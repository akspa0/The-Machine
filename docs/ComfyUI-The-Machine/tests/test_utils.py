from ComfyUI-The-Machine.utils import privacy, manifest

def test_scrub_filename():
    assert privacy.scrub_filename('test.wav') == 'test.wav'

def test_scrub_metadata():
    assert privacy.scrub_metadata({'artist': 'PII'}) == {'artist': 'PII'}

def test_validate_manifest():
    assert manifest.validate_manifest({}) is True

def test_update_manifest():
    m = {'a': 1}
    updates = {'b': 2}
    result = manifest.update_manifest(m, updates)
    assert result['b'] == 2 