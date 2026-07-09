from dataset.china_loader import load_china_dataset

def test_load_dataset():
    data = load_china_dataset()
    assert len(data) > 0
    assert "File" in data[0]
    assert "Effort" in data[0]
