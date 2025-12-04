// parkinsons_detector_with_cadence.cpp
// Optimized detector with enhanced FOG vs Normal Stop discrimination
#include "arm_math.h"
#include "lsm6dsl_reg.h"
#include "mbed.h"
#include "ble/BLE.h"
#include <cmath>
#include <cstring>
#include <cstdio>

// Add this at the top with other variables
bool csv_header_printed = false;

// ARM_MATH_CM4 assumed defined by build system
#define __FPU_PRESENT 1

// --- Config ---
constexpr int SAMPLE_SIZE = 128;     // FFT size (keep current)
constexpr int BUFFER_SIZE = 156;     // circular buffer size (>= SAMPLE_SIZE + margin)

// Dual-window FOG detection
constexpr int FOG_SHORT_WINDOW = 32;
constexpr int FOG_COMPARE_WINDOW = 32;

// Buffers
static float32_t accel_magnitude[BUFFER_SIZE]; // circular buffer holding filtered magnitudes
static float32_t fft_input[SAMPLE_SIZE];       // time-domain window oldest->newest
static float32_t output[SAMPLE_SIZE];          // CMSIS RFFT output (packed)

// LEDs
DigitalOut led_tremor(PB_14, 0);
DigitalOut led_dyskinesia(PA_5, 0);
DigitalOut led_fog(PC_9, 0);

// I2C + serial
I2C i2c(PB_11, PB_10);
BufferedSerial pc(USBTX, USBRX, 115200);

// Sensor
#define LSM6DSL_I2C_ADDR 0x6A
stmdev_ctx_t dev_ctx;

// State machine
enum FogState { IDLE, WALKING, FOG_POSSIBLE, FOG_DETECTED };
static FogState current_state = IDLE;
static int walking_counter = 0;
static int fog_counter = 0;
static int idle_counter = 0;
static float previous_variance = 0.0f;

// Tunables (time-domain variance thresholds are scaled for filtered magnitude values)
constexpr float WALKING_VARIANCE_SHORT = 0.002f; // short-window walking threshold (tune)
constexpr float FOG_VARIANCE_SHORT = 0.001f;     // short-window fog threshold (tune)
constexpr float VARIANCE_RATIO_THRESHOLD = 0.5f; // recent/older ratio threshold (tune)

// Tremor/dyskinesia thresholds (frequency-domain)
constexpr float TREMOR_BIN_THRESHOLD = 2.0f;
constexpr float DYSKINESIA_BIN_THRESHOLD = 3.0f;
constexpr float TREMOR_ENERGY_THRESHOLD = 10.0f;
constexpr float DYSKINESIA_ENERGY_THRESHOLD = 15.0f;

// Step/cadence detection parameters
constexpr float STEP_PEAK_THRESHOLD = 0.02f;    // HP-filtered magnitude peak threshold for a step (tune)
constexpr int STEP_REFRACTORY = 10;            // samples (~200ms) refractory period to avoid double-counting a step
constexpr float CADENCE_WINDOW_SEC = 2.0f;     // seconds to measure recent cadence
constexpr int MIN_STEPS_FOR_FOG = 5;           // require at least this many steps in recent window to allow FOG
constexpr int HISTORY_SIZE = 64;  
constexpr int MIN_WALK_FRAMES = 10; 

// Step-intent detection threshold: small jerk threshold (transient derivative)
constexpr float STEP_INTENT_DERIV_THRESHOLD = 0.025f; // tune

// Debounce / confirmation
constexpr int WALKING_CONFIRM_FRAMES = 3;
constexpr int FOG_CONFIRM_FRAMES = 4;
constexpr int IDLE_GRADUAL_FRAMES = 5;
constexpr int BLE_STABLE_FRAMES = 3;

// BLE UUIDs (unchanged)
const UUID PARKINSONS_SERVICE_UUID("12340000-1234-5678-1234-56789abcdef0");
const UUID TREMOR_CHAR_UUID("12340001-1234-5678-1234-56789abcdef0");
const UUID DYSKINESIA_CHAR_UUID("12340002-1234-5678-1234-56789abcdef0");
const UUID FOG_CHAR_UUID("12340003-1234-5678-1234-56789abcdef0");

// BLE state
bool tremor_state = false;
bool dyskinesia_state = false;
bool fog_state = false;

// --- Walking history ---
static float walking_history[HISTORY_SIZE] = {0};
static int walking_history_index = 0;


ReadOnlyGattCharacteristic<bool> tremorChar(TREMOR_CHAR_UUID, &tremor_state,
                                           GattCharacteristic::BLE_GATT_CHAR_PROPERTIES_NOTIFY);
ReadOnlyGattCharacteristic<bool> dyskinesiaChar(DYSKINESIA_CHAR_UUID, &dyskinesia_state,
                                               GattCharacteristic::BLE_GATT_CHAR_PROPERTIES_NOTIFY);
ReadOnlyGattCharacteristic<bool> fogChar(FOG_CHAR_UUID, &fog_state,
                                        GattCharacteristic::BLE_GATT_CHAR_PROPERTIES_NOTIFY);
GattCharacteristic *characteristics[] = {&tremorChar, &dyskinesiaChar, &fogChar};
GattService parkinsonsService(PARKINSONS_SERVICE_UUID, characteristics, 3);

// Platform I2C wrappers (unchanged)
int32_t platform_write(void *handle, uint8_t reg, const uint8_t *bufp, uint16_t len) {
  char data[17];
  if (len > 16) return -1;
  data[0] = reg;
  memcpy(&data[1], bufp, len);
  return i2c.write(((int)(intptr_t)handle << 1), data, len + 1) == 0 ? 0 : -1;
}

int32_t platform_read(void *handle, uint8_t reg, uint8_t *bufp, uint16_t len) {
  char reg_addr = reg;
  if (i2c.write(((int)(intptr_t)handle << 1), &reg_addr, 1, true) != 0) return -1;
  return i2c.read(((int)(intptr_t)handle << 1), (char *)bufp, len) == 0 ? 0 : -1;
}

// BLE debounce helper
struct BleDebounce {
  bool last_sent = false;
  bool candidate = false;
  int counter = 0;
} ble_tremor, ble_dyskinesia, ble_fog;

void ble_process_channel(BleDebounce &d, bool candidate_value, ReadOnlyGattCharacteristic<bool> &ch, bool &state_var) {
  if (candidate_value != d.candidate) {
    d.candidate = candidate_value;
    d.counter = 1;
  } else {
    if (d.counter > 0) d.counter++;
  }
  if (d.counter >= BLE_STABLE_FRAMES && d.last_sent != d.candidate) {
    d.last_sent = d.candidate;
    state_var = d.candidate;
    BLE::Instance().gattServer().write(ch.getValueHandle(), (uint8_t *)&state_var, sizeof(bool));
  }
}

void update_ble(bool tremor, bool dyskinesia, bool fog) {
  ble_process_channel(ble_tremor, tremor, tremorChar, tremor_state);
  ble_process_channel(ble_dyskinesia, dyskinesia, dyskinesiaChar, dyskinesia_state);
  ble_process_channel(ble_fog, fog, fogChar, fog_state);
}

// Global FFT instance (init once)
arm_rfft_fast_instance_f32 fft_instance;

// Simple high-pass IIR to remove gravity / slow drift from magnitude signal
class HighPass {
public:
  HighPass(float a = 0.98f) : alpha(a), y_prev(0.0f), x_prev(0.0f), init(false) {}
  float filter(float x) {
    if (!init) { x_prev = x; y_prev = 0.0f; init = true; return 0.0f; }
    float y = alpha * (y_prev + x - x_prev);
    y_prev = y;
    x_prev = x;
    return y;
  }
private:
  float alpha;
  float y_prev, x_prev;
  bool init;
};

HighPass hp(0.98f);

// Forward declaration
void analyze_motion(const float *magnitudes, int bins_count, float sampling_rate, const float *time_window);

// Initialize sensor, FFT and run main loop
void test_fft_accelerometer() {
  int16_t data_raw[3];
  int buffer_index = 0;

  printf("Initializing sensor and FFT...\n");

  dev_ctx.write_reg = platform_write;
  dev_ctx.read_reg = platform_read;
  dev_ctx.handle = (void *)LSM6DSL_I2C_ADDR;

  thread_sleep_for(500);

  uint8_t whoami = 0;
  if (lsm6dsl_device_id_get(&dev_ctx, &whoami) != 0 || whoami != 0x6A) {
    printf("Device not found or ID mismatch. WHO_AM_I = 0x%X\n", whoami);
    return;
  }

  lsm6dsl_reset_set(&dev_ctx, PROPERTY_ENABLE);
  uint8_t rst;
  do { lsm6dsl_reset_get(&dev_ctx, &rst); } while (rst);

  // ODR 52Hz, full scale 2g
  lsm6dsl_xl_data_rate_set(&dev_ctx, LSM6DSL_XL_ODR_52Hz);
  lsm6dsl_xl_full_scale_set(&dev_ctx, LSM6DSL_2g);

  // Initialize FFT instance once
  if (arm_rfft_fast_init_f32(&fft_instance, SAMPLE_SIZE) != ARM_MATH_SUCCESS) {
    printf("FFT init failed\n");
    return;
  }

  // Fill buffer initially with a few readings (use HP filter before storing)
  printf("Collecting baseline samples...\n");
  for (int i = 0; i < BUFFER_SIZE; i++) {
    while (true) {
      lsm6dsl_status_reg_t status;
      lsm6dsl_status_reg_get(&dev_ctx, &status);
      if (status.xlda) {
        lsm6dsl_acceleration_raw_get(&dev_ctx, data_raw);
        float x = lsm6dsl_from_fs2g_to_mg(data_raw[0]) / 1000.0f;
        float y = lsm6dsl_from_fs2g_to_mg(data_raw[1]) / 1000.0f;
        float z = lsm6dsl_from_fs2g_to_mg(data_raw[2]) / 1000.0f;
        float mag = sqrtf(x * x + y * y + z * z);
        accel_magnitude[i] = hp.filter(mag);
        break;
      }
      thread_sleep_for(1);
    }
  }

  printf("Starting continuous motion detection (52Hz)...\n");

  // Main acquisition loop
  while (true) {
    lsm6dsl_status_reg_t status;
    lsm6dsl_status_reg_get(&dev_ctx, &status);
    if (status.xlda) {
      lsm6dsl_acceleration_raw_get(&dev_ctx, data_raw);
      float x = lsm6dsl_from_fs2g_to_mg(data_raw[0]) / 1000.0f;
      float y = lsm6dsl_from_fs2g_to_mg(data_raw[1]) / 1000.0f;
      float z = lsm6dsl_from_fs2g_to_mg(data_raw[2]) / 1000.0f;
      float mag = sqrtf(x * x + y * y + z * z);

      // HP filter then store
      float hp_mag = hp.filter(mag);
      accel_magnitude[buffer_index] = hp_mag;
      buffer_index = (buffer_index + 1) % BUFFER_SIZE;

      // Build time-domain window (oldest -> newest)
      for (int i = 0; i < SAMPLE_SIZE; i++) {
        int idx = (buffer_index + BUFFER_SIZE - SAMPLE_SIZE + i) % BUFFER_SIZE;
        fft_input[i] = accel_magnitude[idx];
      }

      // Keep a copy of the un-windowed time-domain data for variance and step analysis
      float time_window[SAMPLE_SIZE];
      for (int i = 0; i < SAMPLE_SIZE; i++) {
        int idx = (buffer_index + BUFFER_SIZE - SAMPLE_SIZE + i) % BUFFER_SIZE;
        time_window[i] = accel_magnitude[idx];
      }

      // Apply Hann window for spectral quality
      for (int n = 0; n < SAMPLE_SIZE; n++) {
        float w = 0.5f * (1.0f - cosf((2.0f * M_PI * n) / (SAMPLE_SIZE - 1)));
        fft_input[n] *= w;
      }

      // Run FFT (output is packed as per CMSIS RFFT)
      arm_rfft_fast_f32(&fft_instance, fft_input, output, 0);

      // Unpack CMSIS RFFT output into magnitudes array for bins 0..N/2 (inclusive)
      const int bins_count = SAMPLE_SIZE / 2 + 1; // 0..N/2 inclusive
      static float magnitudes[ (SAMPLE_SIZE/2) + 1 ];
      // bin 0 (DC)
      magnitudes[0] = fabsf(output[0]);
      // Nyquist bin
      magnitudes[bins_count - 1] = fabsf(output[1]);
      // interior bins
      for (int k = 1; k < bins_count - 1; k++) {
        float real = output[2 * k];
        float imag = output[2 * k + 1];
        magnitudes[k] = sqrtf(real * real + imag * imag);
      }

      // Analyze (pass bins_count and time_window)
      analyze_motion(magnitudes, bins_count, 52.0f, time_window);
    }

    BLE::Instance().processEvents();
    thread_sleep_for(20);
  }
}

// Utility: count recent step-like peaks in time_window (unwindowed), return count
static int count_recent_steps(const float *time_window, float sampling_rate) {
  int window_samples = (int)roundf(CADENCE_WINDOW_SEC * sampling_rate);
  if (window_samples > SAMPLE_SIZE) window_samples = SAMPLE_SIZE;
  int start = SAMPLE_SIZE - window_samples;
  int count = 0;
  int refractory = 0;
  for (int i = start+1; i < SAMPLE_SIZE-1; i++) {
    float v = time_window[i];
    // simple local-peak + threshold detection
    if (refractory > 0) { refractory--; continue; }
    if (v > STEP_PEAK_THRESHOLD && v > time_window[i-1] && v >= time_window[i+1]) {
      count++;
      refractory = STEP_REFRACTORY;
    }
  }
  return count;
}

// Utility: detect step-intent transients (derivative peaks) within recent short window
static int detect_step_intent(const float *time_window) {
  // compute first derivative and count spikes above threshold
  int count = 0;
  for (int i = 1; i < SAMPLE_SIZE; i++) {
    float deriv = fabsf(time_window[i] - time_window[i-1]);
    if (deriv > STEP_INTENT_DERIV_THRESHOLD) count++;
  }
  // we only care if there was at least one strong transient
  return (count > 0) ? count : 0;
}

void analyze_motion(const float *magnitudes, int bins_count, float sampling_rate, const float *time_window) {
    // frequency resolution
    float freq_res = sampling_rate / (float)((bins_count - 1) * 2); // N = 2*(bins_count-1)

    // Frequency-domain detection
    int tremor_count = 0, dyskinesia_count = 0;
    float tremor_energy = 0.0f, dyskinesia_energy = 0.0f, trembling_power = 0.0f;
    int max_bin = bins_count - 1;

    for (int k = 1; k < max_bin; k++) {
        float freq = k * freq_res;
        float amp = magnitudes[k];

        if (freq >= 3.0f && freq <= 5.0f) {
            tremor_energy += amp;
            if (amp >= TREMOR_BIN_THRESHOLD) tremor_count++;
        } else if (freq > 5.0f && freq <= 7.0f) {
            dyskinesia_energy += amp;
            if (amp >= DYSKINESIA_BIN_THRESHOLD) dyskinesia_count++;
        }
        if (freq >= 3.0f && freq <= 8.0f) trembling_power += amp;
    }

    bool tremor_detected = (tremor_count >= 1) || (tremor_energy > TREMOR_ENERGY_THRESHOLD);
    bool dyskinesia_detected = (dyskinesia_count >= 1) || (dyskinesia_energy > DYSKINESIA_ENERGY_THRESHOLD);
    led_tremor = tremor_detected ? 1 : 0;
    led_dyskinesia = dyskinesia_detected ? 1 : 0;

    // Time-domain variance windows for FOG
    float recent_mean = 0.0f, recent_variance = 0.0f;
    for (int i = SAMPLE_SIZE - FOG_SHORT_WINDOW; i < SAMPLE_SIZE; i++) recent_mean += time_window[i];
    recent_mean /= FOG_SHORT_WINDOW;
    for (int i = SAMPLE_SIZE - FOG_SHORT_WINDOW; i < SAMPLE_SIZE; i++) {
        float d = time_window[i] - recent_mean;
        recent_variance += d * d;
    }
    recent_variance /= FOG_SHORT_WINDOW;

    float older_mean = 0.0f, older_variance = 0.0f;
    int older_start = SAMPLE_SIZE - FOG_SHORT_WINDOW - FOG_COMPARE_WINDOW;
    for (int i = older_start; i < older_start + FOG_COMPARE_WINDOW; i++) older_mean += time_window[i];
    older_mean /= FOG_COMPARE_WINDOW;
    for (int i = older_start; i < older_start + FOG_COMPARE_WINDOW; i++) {
        float d = time_window[i] - older_mean;
        older_variance += d * d;
    }
    older_variance /= FOG_COMPARE_WINDOW;

    float variance_ratio = (older_variance > 1e-8f) ? (recent_variance / older_variance) : 1.0f;

    // --- Update walking history ---
    walking_history[walking_history_index] = recent_variance;
    walking_history_index = (walking_history_index + 1) % HISTORY_SIZE;

    int recent_walking_frames = 0;
    for (int i = 0; i < HISTORY_SIZE; i++)
        if (walking_history[i] > WALKING_VARIANCE_SHORT) recent_walking_frames++;
    bool sufficient_recent_walking = (recent_walking_frames >= MIN_WALK_FRAMES);

    // Steps & intent
    int recent_steps = count_recent_steps(time_window, sampling_rate);
    int step_intent_count = detect_step_intent(time_window);
    float recent_cadence = recent_steps / (float)CADENCE_WINDOW_SEC;

    bool fog_detected = false;

    // --- Enhanced state machine ---
    switch (current_state) {
        case IDLE: // IDLE
            if (recent_variance > WALKING_VARIANCE_SHORT) {
                walking_counter++;
                if (walking_counter >= WALKING_CONFIRM_FRAMES) {
                    current_state = WALKING; // WALKING
                    walking_counter = 0;
                    printf("State -> WALKING\n");
                }
            } else walking_counter = 0;
            led_fog = 1;
            break;

        case WALKING: // WALKING
        {
            bool sudden_stop = (older_variance > WALKING_VARIANCE_SHORT*0.5f) &&
                               (recent_variance < FOG_VARIANCE_SHORT) &&
                               (variance_ratio < VARIANCE_RATIO_THRESHOLD);
            bool trembling_ok = (trembling_power > TREMOR_ENERGY_THRESHOLD*0.5f);
            bool intent_ok = (step_intent_count > 0);

            if ( (sudden_stop && sufficient_recent_walking && (trembling_ok || intent_ok)) ||
                 (recent_variance < FOG_VARIANCE_SHORT && sufficient_recent_walking && (trembling_ok || intent_ok)) ) {
                current_state = FOG_POSSIBLE;
                fog_counter = 0;
                idle_counter = 0;
                printf("State -> FOG_POSSIBLE\n");
            } else if (recent_variance < FOG_VARIANCE_SHORT && older_variance < (WALKING_VARIANCE_SHORT*0.5f)) {
                idle_counter++;
                if (idle_counter >= IDLE_GRADUAL_FRAMES) {
                    current_state = IDLE; // IDLE
                    idle_counter = 0;
                    printf("State -> IDLE (gradual stop)\n");
                }
            } else idle_counter = 0;

            led_fog = 1;
        }
        break;

        case FOG_POSSIBLE: // FOG_POSSIBLE
        {
            bool trembling_ok = (trembling_power > TREMOR_ENERGY_THRESHOLD*0.5f);
            bool intent_ok = (step_intent_count > 0);

            if (recent_variance < FOG_VARIANCE_SHORT && (trembling_ok || intent_ok)) {
                fog_counter++;
                if (fog_counter >= FOG_CONFIRM_FRAMES) {
                    current_state = FOG_DETECTED; // FOG_DETECTED
                    fog_detected = true;
                    led_fog = 0;
                    printf("State -> FOG_DETECTED\n");
                }
            } else if (recent_variance > WALKING_VARIANCE_SHORT) {
                current_state = WALKING; // WALKING
                fog_counter = 0;
                printf("State -> WALKING (resumed)\n");
            }
        }
        break;

        case FOG_DETECTED: // FOG_DETECTED
            fog_detected = true;
            led_fog = 0;
            if (recent_variance > WALKING_VARIANCE_SHORT) {
                current_state = WALKING; // WALKING
                led_fog = 1;
                fog_counter = 0;
                printf("State -> WALKING (FOG ended)\n");
            }
            break;
    }

    previous_variance = recent_variance;

    // BLE update
    update_ble(tremor_detected, dyskinesia_detected, fog_detected);
}

// BLE and utility functions (unchanged)
void onBleInitError(BLE &ble, ble_error_t error) {
  printf("BLE Init failed: %d\n", error);
}

void printMacAddress() {
  BLE &ble = BLE::Instance();
  ble::own_address_type_t addrType;
  ble::address_t address;
  ble.gap().getAddress(addrType, address);
  printf("MAC Address: %02X:%02X:%02X:%02X:%02X:%02X\n",
         address[5], address[4], address[3], address[2], address[1], address[0]);
}

void bleInitComplete(BLE::InitializationCompleteCallbackContext *params) {
  BLE &ble = params->ble;
  ble_error_t error = params->error;
  if (error != BLE_ERROR_NONE) { onBleInitError(ble, error); return; }

  printMacAddress();

  ble.gattServer().addService(parkinsonsService);

  ble::AdvertisingParameters adv_params(ble::advertising_type_t::CONNECTABLE_UNDIRECTED,
                                        ble::adv_interval_t(ble::millisecond_t(1000)));

  static uint8_t adv_buffer[ble::LEGACY_ADVERTISING_MAX_SIZE];
  ble::AdvertisingDataBuilder adv_data_builder(adv_buffer);

  adv_data_builder.setFlags(ble::adv_data_flags_t::BREDR_NOT_SUPPORTED |
                            ble::adv_data_flags_t::LE_GENERAL_DISCOVERABLE);
  adv_data_builder.setName("ParkinsonsDetector");
  adv_data_builder.setLocalServiceList(mbed::make_Span(&PARKINSONS_SERVICE_UUID, 1));

  ble.gap().setAdvertisingParameters(ble::LEGACY_ADVERTISING_HANDLE, adv_params);
  ble.gap().setAdvertisingPayload(ble::LEGACY_ADVERTISING_HANDLE, adv_data_builder.getAdvertisingData());
  ble.gap().startAdvertising(ble::LEGACY_ADVERTISING_HANDLE);

  printf("BLE Initialized and Advertising...\n");
}

int main() {
  thread_sleep_for(1000);
  printf("System Start\n");

  BLE &ble = BLE::Instance();
  ble.init(bleInitComplete);
  printf("Initializing BLE...\n");

  while (!ble.hasInitialized()) {
    ble.processEvents();
    thread_sleep_for(10);
  }
  printf("BLE Initialized\n");

  test_fft_accelerometer();

  return 0;
}


