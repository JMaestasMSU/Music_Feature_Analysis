from preprocessing.feature_extraction import extract_features


def test_extract_features_returns_dict():
    feats = extract_features('dummy/path.wav')
    assert isinstance(feats, dict)
    assert 'mfcc' in feats
