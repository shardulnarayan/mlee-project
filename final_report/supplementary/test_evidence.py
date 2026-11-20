"""Run: python -m unittest final_report.supplementary.test_evidence."""
import math
import tempfile
import unittest
import importlib.util
from pathlib import Path
from .evidence import CLASSES, assemble_tables, manifest, mean_sd, read_csv, scores, write_csv

class EvidenceTests(unittest.TestCase):
    def test_perfect(self):
        result, _ = scores(CLASSES, CLASSES)
        self.assertEqual(result['macro_f1'], 1)
        self.assertEqual(result['accuracy'], 1)

    def test_absent_classes(self):
        result, _ = scores(['Healthy'], ['Healthy'])
        self.assertEqual(result['macro_f1'], 0.2)

    def test_minority_error(self):
        _, rows = scores(['Leaf_Miner', 'Healthy'], ['Healthy', 'Healthy'])
        self.assertEqual(next(r['recall'] for r in rows if r['class_name'] == 'Leaf_Miner'), 0)
        self.assertEqual(next(r['precision'] for r in rows if r['class_name'] == 'Healthy'), 0.5)

    def test_invalid(self):
        for a, b in [([], []), (CLASSES, []), (['unknown'], ['Healthy'])]:
            with self.assertRaises(ValueError): scores(a, b)

    def test_sample_sd(self):
        self.assertEqual(mean_sd([1, 2, 3]), (2, 1))
        self.assertEqual(mean_sd([1]), (1, None))
        with self.assertRaises(ValueError): mean_sd([math.nan])

    def test_manifest(self):
        rows = manifest()
        self.assertEqual(len(rows), 1004)
        self.assertEqual(sum(r['split'] == 'test' for r in rows), 151)

    def test_no_fabricated_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            assemble_tables(tmp)
            self.assertFalse((Path(tmp)/'table6_seeded_summary.csv').exists())
            self.assertFalse((Path(tmp)/'table7_attention_summary.csv').exists())
            self.assertTrue((Path(tmp)/'table9_existing_group_cv.csv').exists())

    def test_seed_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_csv(Path(tmp)/'seeded_runs.csv', [dict(variant='raw', seed=i,
                best_val_accuracy=v, accuracy=v, macro_f1=v, mean_train_seconds=10)
                for i, v in enumerate([0.6, 0.8, 1.0])])
            assemble_tables(tmp)
            row=next(r for r in read_csv(Path(tmp)/'table6_seeded_summary.csv') if r['metric']=='accuracy')
            self.assertAlmostEqual(float(row['mean']), 0.8)
            self.assertAlmostEqual(float(row['sample_sd']), 0.2)

@unittest.skipUnless(importlib.util.find_spec('PIL'), 'Pillow unavailable')
class PixelFaultTests(unittest.TestCase):
    def test_brightness_clipping_without_mutating_source(self):
        from PIL import Image
        from .pixel_faults import corrupt_pixels
        source = Image.new('RGB', (224, 224), (200, 100, 20))
        result = corrupt_pixels(source, 'brightness_factor', 2)
        self.assertEqual(result.getpixel((0, 0)), (255, 200, 40))
        self.assertEqual(source.getpixel((0, 0)), (200, 100, 20))

    def test_blur_reduces_isolated_detail(self):
        from PIL import Image
        from .pixel_faults import corrupt_pixels
        source = Image.new('RGB', (224, 224), 'black')
        source.putpixel((112, 112), (255, 255, 255))
        result = corrupt_pixels(source, 'gaussian_blur_radius', 3)
        self.assertLess(result.getpixel((112, 112))[0], 255)
        self.assertGreater(result.getpixel((112, 112))[0], 0)

    def test_jpeg_rotation_and_settings(self):
        from PIL import Image
        from .pixel_faults import corrupt_pixels, corruption_cases
        source = Image.new('RGB', (224, 224), 'white')
        jpeg = corrupt_pixels(source, 'jpeg_quality', 5)
        self.assertEqual(jpeg.size, (224, 224))
        self.assertEqual(jpeg.mode, 'RGB')
        rotated = corrupt_pixels(source, 'rotation_degrees', 20)
        self.assertEqual(rotated.getpixel((0, 0)), (0, 0, 0))
        self.assertIn(('jpeg_quality', 5), corruption_cases())
        with self.assertRaises(ValueError): corrupt_pixels(source, 'unknown', 1)

if __name__ == '__main__': unittest.main()
