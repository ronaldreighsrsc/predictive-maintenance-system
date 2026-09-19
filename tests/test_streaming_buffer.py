import pytest
import numpy as np
import concurrent.futures
from src.preprocessing.streaming_window_buffer import EquipmentSlidingBuffer


def test_buffer_initialization_and_shape():
    buffer = EquipmentSlidingBuffer(window_size=20)
    raw_reading = [85.0, 45.0, 0.5, 1800.0, 12.0, 75.0, 200.0]
    features = buffer.push_reading("CAEX-101", raw_reading, operating_hours=500.0)

    assert isinstance(features, np.ndarray)
    assert features.shape == (1, 68)
    assert features.dtype == np.float32


def test_buffer_dict_input():
    buffer = EquipmentSlidingBuffer(window_size=20)
    reading_dict = {
        'engine_temp': 88.0,
        'oil_pressure': 42.0,
        'vibration_level': 0.6,
        'rpm': 1750.0,
        'fuel_consumption': 11.5,
        'coolant_temp': 78.0,
        'hydraulic_pressure': 195.0
    }
    features = buffer.push_reading("CAEX-102", reading_dict, operating_hours=120.0)
    assert features.shape == (1, 68)
    assert features[0, 0] == pytest.approx(88.0)
    assert features[0, 7] == pytest.approx(120.0)  # operating_hours


def test_buffer_warmup_and_status():
    buffer = EquipmentSlidingBuffer(window_size=20)
    eq_id = "CAEX-103"
    raw_reading = [85.0, 45.0, 0.5, 1800.0, 12.0, 75.0, 200.0]

    # Primera lectura: no está warmed up
    buffer.push_reading(eq_id, raw_reading, operating_hours=1.0)
    status = buffer.get_buffer_status(eq_id)
    assert status["is_warmed_up"] is False
    assert status["buffer_length"] == 1

    # Ingestar 19 lecturas más
    for i in range(2, 21):
        buffer.push_reading(eq_id, raw_reading, operating_hours=float(i))

    status = buffer.get_buffer_status(eq_id)
    assert status["is_warmed_up"] is True
    assert status["buffer_length"] == 20
    assert status["total_ingested_cycles"] == 20


def test_buffer_cross_sensor_calculations():
    buffer = EquipmentSlidingBuffer(window_size=20)
    eq_id = "CAEX-104"
    # engine_temp=90, oil_pressure=45 -> temp_oil_ratio = 2.0
    # coolant_temp=80 -> temp_coolant_diff = 10.0
    # rpm=2000, vibration=1.0 -> vib_per_rpm = 1.0 / 2.0 = 0.5
    # fuel=14.0 -> fuel_eff = 14.0 / 2.0 = 7.0
    reading = [90.0, 45.0, 1.0, 2000.0, 14.0, 80.0, 210.0]
    features = buffer.push_reading(eq_id, reading, operating_hours=50.0)

    # Las últimas 4 features son los ratios cross-sensor
    temp_oil_ratio = features[0, -4]
    temp_coolant_diff = features[0, -3]
    vib_per_rpm = features[0, -2]
    fuel_efficiency = features[0, -1]

    assert temp_oil_ratio == pytest.approx(2.0, rel=1e-3)
    assert temp_coolant_diff == pytest.approx(10.0, rel=1e-3)
    assert vib_per_rpm == pytest.approx(0.5, rel=1e-3)
    assert fuel_efficiency == pytest.approx(7.0, rel=1e-3)


def test_buffer_concurrency_thread_safety():
    buffer = EquipmentSlidingBuffer(window_size=20)
    raw_reading = [85.0, 45.0, 0.5, 1800.0, 12.0, 75.0, 200.0]

    def worker(truck_id: int):
        eq_name = f"CAEX-{truck_id:03d}"
        for c in range(30):
            buffer.push_reading(eq_name, raw_reading, operating_hours=float(c))
        return eq_name

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        truck_ids = list(range(20))
        results = list(executor.map(worker, truck_ids))

    assert len(results) == 20
    assert len(buffer.get_active_equipments()) == 20
    for eq in buffer.get_active_equipments():
        status = buffer.get_buffer_status(eq)
        assert status["is_warmed_up"] is True
        assert status["total_ingested_cycles"] == 30
