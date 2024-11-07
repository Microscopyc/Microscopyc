from Image_identifier.image_identifier import (
    Coccus, Bacillus, Spiral, BacteriaShapeClassifier)


# Testing Coccus classification
def test_coccus_classify():
    coccus = Coccus()

    assert coccus.classify(1.0) == "Coccus"  # Should classify "Coccus"
    assert coccus.classify(0.9) == "Coccus"   # Edge of Coccus range
    assert coccus.classify(1.1) == "Coccus"   # Edge of Coccus range
    assert coccus.classify(0.8) is None  # Should return None
    assert coccus.classify(1.2) is None


# Testing Bacillus classification
def test_bacillus_classify():
    bacillus = Bacillus()

    assert bacillus.classify(0.5) == "Bacillus"   # Should classify "Bacillus"
    assert bacillus.classify(0.2) == "Bacillus"   # Edge of Bacillus range
    assert bacillus.classify(0.8) == "Bacillus"   # Edge of Bacillus range
    assert (bacillus.classify(0.1) is None)
    assert (bacillus.classify(0.9) is None)
    #  assert bacillus.classify(0.1) == "Bacillus"   # Should raise an error


# Testing Spiral classification
def test_spiral_classify():
    spiral = Spiral()

    assert spiral.classify(0.1) == "Spiral"  # Should classify "Spiral"
    assert spiral.classify(0.3) is None
    assert spiral.classify(0.19) == "Spiral"  # Edge of Spiral range
    assert spiral.classify(0.2) is None


def test_bacteria_shape_classifier():
    classifier = BacteriaShapeClassifier()

    # Test with Coccus strategy
    assert (
        classifier.classify_shape(1.0) == "Coccus"
    )  # Aspect ratio in Coccus range
    assert (
        classifier.classify_shape(0.8) == "Bacillus"
    )  # Aspect ratio in Bacillus range
    assert (
        classifier.classify_shape(0.15) == "Spiral"
    )  # Aspect ratio in Spiral range
    assert (
        classifier.classify_shape(1.2) == "unknown, the ratio is 1.2"
    )  # Aspect ratio doesn't match any strategy
