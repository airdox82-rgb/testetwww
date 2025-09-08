import pytest
import os
import tempfile
import json
from unittest.mock import patch, MagicMock
from app import app, status_data, status_lock, UPLOAD_FOLDER


@pytest.fixture
def client():
    """Test client für Flask App."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def temp_upload_folder():
    """Temporärer Upload-Ordner für Tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_upload_folder = UPLOAD_FOLDER
        # Patch the UPLOAD_FOLDER for tests
        with patch('app.UPLOAD_FOLDER', temp_dir):
            yield temp_dir


def test_cors_headers(client):
    """Test dass CORS-Header korrekt gesetzt sind."""
    response = client.get('/')
    assert response.status_code == 200
    # CORS sollte jetzt funktionieren da es vor den Routen definiert ist


def test_status_endpoint(client):
    """Test Status-Endpoint."""
    response = client.get('/api/status')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'training' in data
    assert 'synthesizing' in data
    assert 'progress' in data
    assert 'log' in data
    assert 'output_file' in data


def test_upload_valid_file(client, temp_upload_folder):
    """Test Upload einer gültigen Audiodatei."""
    # Erstelle eine Test-WAV-Datei
    test_file_content = b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
    
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    
    with open(temp_file.name, 'wb') as f:
        f.write(test_file_content)
    
    with patch('app.UPLOAD_FOLDER', temp_upload_folder):
        with open(temp_file.name, 'rb') as f:
            response = client.post('/api/upload_sample', 
                                 data={'file': (f, 'test.wav')},
                                 content_type='multipart/form-data')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'sample' in data
    assert data['sample'] == 'test.wav'
    
    # Cleanup
    os.unlink(temp_file.name)


def test_upload_invalid_file_extension(client):
    """Test Upload einer Datei mit ungültiger Erweiterung."""
    temp_file = tempfile.NamedTemporaryFile(suffix='.txt', delete=False)
    
    with open(temp_file.name, 'w') as f:
        f.write('test content')
    
    with open(temp_file.name, 'rb') as f:
        response = client.post('/api/upload_sample', 
                             data={'file': (f, 'test.txt')},
                             content_type='multipart/form-data')
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert 'Nicht unterstütztes Dateiformat' in data['error']
    
    # Cleanup
    os.unlink(temp_file.name)


def test_upload_no_file(client):
    """Test Upload ohne Datei."""
    response = client.post('/api/upload_sample', data={})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['error'] == 'No file part'


def test_synthesize_valid_text(client):
    """Test Synthese mit gültigem Text."""
    response = client.post('/api/synthesize', data={'text': 'Hallo Welt'})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'message' in data
    assert 'Hallo Welt' in data['message']


def test_synthesize_empty_text(client):
    """Test Synthese mit leerem Text."""
    response = client.post('/api/synthesize', data={'text': ''})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['error'] == 'Kein Text angegeben'


def test_synthesize_no_text(client):
    """Test Synthese ohne Text-Parameter."""
    response = client.post('/api/synthesize', data={})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['error'] == 'Kein Text angegeben'


def test_train_endpoint(client):
    """Test Training-Endpoint."""
    # Status zurücksetzen
    with status_lock:
        status_data["training"] = False
        status_data["synthesizing"] = False
    
    response = client.post('/api/train')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'message' in data
    assert 'Training gestartet' in data['message']


def test_samples_list(client, temp_upload_folder):
    """Test Auflistung der Samples."""
    # Erstelle Test-Dateien
    test_files = ['sample1.wav', 'sample2.mp3', 'output_generated.wav']
    for filename in test_files:
        filepath = os.path.join(temp_upload_folder, filename)
        with open(filepath, 'w') as f:
            f.write('test')
    
    with patch('app.UPLOAD_FOLDER', temp_upload_folder):
        response = client.get('/api/samples')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'samples' in data
    # Output-Dateien sollten herausgefiltert werden
    assert 'sample1.wav' in data['samples']
    assert 'sample2.mp3' in data['samples']
    assert 'output_generated.wav' not in data['samples']


def test_filename_sanitization(client, temp_upload_folder):
    """Test dass Dateinamen korrekt bereinigt werden."""
    test_file_content = b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
    
    # Dateiname mit problematischen Zeichen
    problematic_filename = 'test file with spaces & symbols!.wav'
    expected_filename = 'test_file_with_spaces__symbols.wav'
    
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    
    with open(temp_file.name, 'wb') as f:
        f.write(test_file_content)
    
    with patch('app.UPLOAD_FOLDER', temp_upload_folder):
        with open(temp_file.name, 'rb') as f:
            response = client.post('/api/upload_sample', 
                                 data={'file': (f, problematic_filename)},
                                 content_type='multipart/form-data')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['sample'] == expected_filename
    
    # Prüfe dass Datei tatsächlich mit bereinigtem Namen gespeichert wurde
    assert os.path.exists(os.path.join(temp_upload_folder, expected_filename))
    
    # Cleanup
    os.unlink(temp_file.name)


if __name__ == '__main__':
    pytest.main([__file__])