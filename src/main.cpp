#include "arm_math.h"
#include "lsm6dsl_reg.h"
#include "mbed.h"
#include "ble/BLE.h"

// ARM_MATH_CM4 is already defined by the build system
#define __FPU_PRESENT 1

constexpr int SAMPLE_SIZE = 128;
constexpr int BUFFER_SIZE = 156;

static float32_t accel_magnitude[BUFFER_SIZE];
static float32_t fft_input[SAMPLE_SIZE];
static float32_t output[2 * SAMPLE_SIZE];

DigitalOut led_tremor(PB_14, 0);  // Start OFF
DigitalOut led_dyskinesia(PA_5, 0);  // Start OFF
DigitalOut led_fog(PC_9, 0);  // Start OFF

I2C i2c(PB_11, PB_10);
BufferedSerial pc(USBTX, USBRX, 115200);

#define LSM6DSL_I2C_ADDR 0x6A
stmdev_ctx_t dev_ctx;

enum FogState { IDLE, WALKING, FOG_POSSIBLE, FOG_DETECTED };
FogState current_state = IDLE;
int walking_counter = 0;
int fog_counter = 0;
float previous_variance = 0.0f;
float baseline_variance = 65.0f;  // Typical idle variance observed
constexpr float WALKING_VARIANCE_THRESHOLD = 100.0f;  // Require significant movement (will show ~1000000)
constexpr float FOG_VARIANCE_THRESHOLD = 70.0f;       // Still/frozen state (will show ~700000)
constexpr float FOG_ENERGY_THRESHOLD = 5.0f;          // Lower threshold

// BLE UUIDs
const UUID PARKINSONS_SERVICE_UUID("12340000-1234-5678-1234-56789abcdef0");
const UUID TREMOR_CHAR_UUID("12340001-1234-5678-1234-56789abcdef0");
const UUID DYSKINESIA_CHAR_UUID("12340002-1234-5678-1234-56789abcdef0");
const UUID FOG_CHAR_UUID("12340003-1234-5678-1234-56789abcdef0");

bool tremor_state = false;
bool dyskinesia_state = false;
bool fog_state = false;

ReadOnlyGattCharacteristic<bool> tremorChar(TREMOR_CHAR_UUID, &tremor_state,
                                             GattCharacteristic::BLE_GATT_CHAR_PROPERTIES_NOTIFY);
ReadOnlyGattCharacteristic<bool> dyskinesiaChar(DYSKINESIA_CHAR_UUID, &dyskinesia_state,
                                                 GattCharacteristic::BLE_GATT_CHAR_PROPERTIES_NOTIFY);
ReadOnlyGattCharacteristic<bool> fogChar(FOG_CHAR_UUID, &fog_state,
                                          GattCharacteristic::BLE_GATT_CHAR_PROPERTIES_NOTIFY);

GattCharacteristic *characteristics[] = {&tremorChar, &dyskinesiaChar, &fogChar};
GattService parkinsonsService(PARKINSONS_SERVICE_UUID, characteristics, 3);

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

void update_ble(bool tremor, bool dyskinesia, bool fog) {
  BLE &ble = BLE::Instance();
  if (tremor != tremor_state) {
    tremor_state = tremor;
    ble.gattServer().write(tremorChar.getValueHandle(), (uint8_t *)&tremor_state, sizeof(bool));
  }
  if (dyskinesia != dyskinesia_state) {
    dyskinesia_state = dyskinesia;
    ble.gattServer().write(dyskinesiaChar.getValueHandle(), (uint8_t *)&dyskinesia_state, sizeof(bool));
  }
  if (fog != fog_state) {
    fog_state = fog;
    ble.gattServer().write(fogChar.getValueHandle(), (uint8_t *)&fog_state, sizeof(bool));
  }
}

void analyze_motion(const float *magnitudes, int sample_size, float sampling_rate);

void test_fft_accelerometer() {
  int16_t data_raw[3];
  int buffer_index = 0;

  printf("Initializing FFT on live accelerometer data...\n");

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
  do {
    lsm6dsl_reset_get(&dev_ctx, &rst);
  } while (rst);

  lsm6dsl_xl_data_rate_set(&dev_ctx, LSM6DSL_XL_ODR_52Hz);
  lsm6dsl_xl_full_scale_set(&dev_ctx, LSM6DSL_2g);

  // Fill buffer with initial readings
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
        accel_magnitude[i] = sqrtf(x * x + y * y + z * z);
        break;
      }
      thread_sleep_for(1);
    }
  }

  printf("Starting continuous motion detection (52Hz)...\n");

  while (true) {
    lsm6dsl_status_reg_t status;
    lsm6dsl_status_reg_get(&dev_ctx, &status);
    if (status.xlda) {
      lsm6dsl_acceleration_raw_get(&dev_ctx, data_raw);
      float x = lsm6dsl_from_fs2g_to_mg(data_raw[0]) / 1000.0f;
      float y = lsm6dsl_from_fs2g_to_mg(data_raw[1]) / 1000.0f;
      float z = lsm6dsl_from_fs2g_to_mg(data_raw[2]) / 1000.0f;
      float magnitude = sqrtf(x * x + y * y + z * z);

      accel_magnitude[buffer_index] = magnitude;
      buffer_index = (buffer_index + 1) % BUFFER_SIZE;

      for (int i = 0; i < SAMPLE_SIZE; i++) {
        int index = (buffer_index + BUFFER_SIZE - SAMPLE_SIZE + i) % BUFFER_SIZE;
        fft_input[i] = accel_magnitude[index];
      }

      arm_rfft_fast_instance_f32 fft_instance;
      if (arm_rfft_fast_init_f32(&fft_instance, SAMPLE_SIZE) != ARM_MATH_SUCCESS) {
        printf("FFT initialization failed.\n");
        return;
      }

      arm_rfft_fast_f32(&fft_instance, fft_input, output, 0);

      float magnitudes[SAMPLE_SIZE / 2];
      for (int i = 0; i < SAMPLE_SIZE / 2; i++) {
        float real = output[2 * i];
        float imag = output[2 * i + 1];
        magnitudes[i] = sqrtf(real * real + imag * imag);
      }

      analyze_motion(magnitudes, SAMPLE_SIZE, 52.0f);
    }

    BLE::Instance().processEvents();
    thread_sleep_for(20);
  }
}

void analyze_motion(const float *magnitudes, int sample_size, float sampling_rate) {
  float frequency_resolution = sampling_rate / sample_size;

  // Lower thresholds for easier detection
  float tremor_threshold = 10.0f;
  float dyskinesia_threshold = 20.0f;

  int tremor_count = 0;
  int dyskinesia_count = 0;
  float tremor_energy = 0.0f;
  float dyskinesia_energy = 0.0f;
  float fog_energy = 0.0f;

  // Calculate variance using detrended data (remove gravity baseline)
  float mean = 0.0f;
  float variance = 0.0f;
  for (int i = 0; i < sample_size; i++) {
    mean += fft_input[i];
  }
  mean /= sample_size;

  // Remove mean (gravity component) and calculate variance
  for (int i = 0; i < sample_size; i++) {
    float detrended = fft_input[i] - mean;
    variance += detrended * detrended;
  }
  variance /= sample_size;

  for (int i = 1; i < sample_size / 2; i++) {
    float freq = i * frequency_resolution;
    float amp = magnitudes[i];

    if (freq >= 3.0f && freq <= 5.0f) {
      tremor_energy += amp;
      if (amp >= tremor_threshold) {
        tremor_count++;
      }
    } else if (freq > 5.0f && freq <= 7.0f) {
      dyskinesia_energy += amp;
      if (amp >= dyskinesia_threshold) {
        dyskinesia_count++;
      }
    }

    if (freq >= 3.0f && freq <= 8.0f) {
      fog_energy += amp;
    }
  }

  // More sensitive detection - use either bin count OR energy
  bool tremor_detected = (tremor_count >= 1) || (tremor_energy > 30.0f);
  bool dyskinesia_detected = (dyskinesia_count >= 1) || (dyskinesia_energy > 40.0f);

  if (tremor_detected) {
    led_tremor = 1;
    led_dyskinesia = 0;
    printf("[TREMOR DETECTED] bins=%d\n", tremor_count);
  } else if (dyskinesia_detected) {
    led_tremor = 1;
    led_dyskinesia = 1;
    printf("[DYSKINESIA DETECTED] bins=%d\n", dyskinesia_count);
  } else {
    led_tremor = 0;
    led_dyskinesia = 0;
  }

  static int debug_counter = 0;
  if (++debug_counter >= 100) {  // Only print every 100 samples (~2 seconds)
    printf("T:%d/%d D:%d/%d V:%d State:%d\n",
           tremor_count, (int)tremor_energy,
           dyskinesia_count, (int)dyskinesia_energy,
           (int)(variance * 10000),
           (int)current_state);
    debug_counter = 0;
  }

  bool fog_detected = false;

  // Detect sudden drop in variance (FOG event)
  float variance_drop = previous_variance - variance;
  bool sudden_drop = (variance_drop > 30.0f) && (variance < baseline_variance + 10.0f);

  switch (current_state) {
    case IDLE:
      if (variance > WALKING_VARIANCE_THRESHOLD) {
        walking_counter++;
        if (walking_counter > 2) {
          current_state = WALKING;
          walking_counter = 0;
          printf("State: WALKING\n");
        }
      } else {
        walking_counter = 0;
      }
      led_fog = 0;
      break;

    case WALKING:
      // Check for sudden drop to baseline (FOG event)
      if (sudden_drop || variance < FOG_VARIANCE_THRESHOLD) {
        current_state = FOG_POSSIBLE;
        fog_counter = 0;
        printf("State: FOG_POSSIBLE (drop:%d)\n", (int)variance_drop);
      }
      led_fog = 0;
      break;

    case FOG_POSSIBLE:
      if (variance < FOG_VARIANCE_THRESHOLD) {
        fog_counter++;
        if (fog_counter > 2) {  // Faster detection
          current_state = FOG_DETECTED;
          printf("State: FOG_DETECTED\n");
        }
      } else if (variance > WALKING_VARIANCE_THRESHOLD) {
        current_state = WALKING;
        printf("State: WALKING (resumed)\n");
      }
      led_fog = 0;
      break;

    case FOG_DETECTED:
      led_fog = 1;
      fog_detected = true;
      if (variance > WALKING_VARIANCE_THRESHOLD) {
        current_state = WALKING;
        led_fog = 0;
        printf("State: WALKING (FOG ended)\n");
      }
      break;
  }

  previous_variance = variance;

  static int tp_counter = 0;

  // Print more frequently for smooth Teleplot curves
  if (++tp_counter >= 5) {
      printf(">Tremor:%d\n", int(tremor_detected));
      printf(">Dyskinesia:%d\n", int(dyskinesia_detected));
      printf(">Variance:%d\n", (int)variance);
      printf(">State:%d\n", (int)current_state);
      tp_counter = 0;
  }

  update_ble(tremor_detected, dyskinesia_detected, fog_detected);
}

void onBleInitError(BLE &ble, ble_error_t error) {
  printf("BLE Init failed: %d\n", error);
}

void printMacAddress() {
  BLE &ble = BLE::Instance();
  ble::own_address_type_t addrType;
  ble::address_t address;
  ble.gap().getAddress(addrType, address);
  printf("MAC Address: %02X:%02X:%02X:%02X:%02X:%02X\n", address[5], address[4], address[3], address[2], address[1], address[0]);
}

void bleInitComplete(BLE::InitializationCompleteCallbackContext *params) {
  BLE &ble = params->ble;
  ble_error_t error = params->error;

  if (error != BLE_ERROR_NONE) {
    onBleInitError(ble, error);
    return;
  }

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

  while (ble.hasInitialized() == false) {
    ble.processEvents();
    thread_sleep_for(10);
  }
  printf("BLE Initialized\n");
  test_fft_accelerometer();

  return 0;
}
