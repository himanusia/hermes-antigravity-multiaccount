"""Short-delivery detection: the worker billed fewer tokens than the prompt assembled.

The fixture pairs are real measurements from the agy route (gemini-3.8-flash,
provider antigravity-subscription-directsdk, 2026-10-12) reported in
soyelmismo/hermes-antigravity-subscription#29, so a regression here reproduces
on the route rather than only in the test.
"""

import os
import unittest
from unittest import mock

from delivery import (
    DISABLE_ENV,
    DeliverySample,
    DeliveryWatch,
    ShortDelivery,
)


class TestDeliveryWatch(unittest.TestCase):
    # A 64,147-character dense-numeric prompt delivered 95,990 tokens whole;
    # doubling it to 126,789 characters delivered 96,651 -- the tail was gone.
    MEASURED_TRUNCATION = [(64_147, 95_990), (126_789, 96_651)]
    # The same series on prose material, where delivery kept pace.
    MEASURED_WHOLE = [(6_211, 49_285), (20_469, 70_601), (34_687, 91_838)]

    def test_flat_tokens_under_doubled_prompt_warns(self):
        watch = DeliveryWatch()
        first, second = self.MEASURED_TRUNCATION
        self.assertIsNone(watch.observe(*first))
        finding = watch.observe(*second)
        if not isinstance(finding, ShortDelivery):
            self.fail(f"expected a ShortDelivery finding, got {finding!r}")
        self.assertEqual(finding.previous, DeliverySample(*first))
        self.assertEqual(finding.sample, DeliverySample(*second))
        self.assertIn("delivered", finding.describe())
        self.assertIn("#29", finding.describe())

    def test_healthy_growth_is_silent(self):
        watch = DeliveryWatch()
        for sample in self.MEASURED_WHOLE:
            self.assertIsNone(watch.observe(*sample))

    def test_first_sample_alone_never_warns(self):
        self.assertIsNone(DeliveryWatch().observe(300_000, 90_000))

    def test_small_prompts_are_ignored(self):
        watch = DeliveryWatch()
        self.assertIsNone(watch.observe(4_000, 41_000))
        self.assertIsNone(watch.observe(12_000, 41_100))

    def test_token_count_falling_while_prompt_grows_warns(self):
        watch = DeliveryWatch()
        self.assertIsNone(watch.observe(100_000, 114_794))
        self.assertIsInstance(watch.observe(252_156, 96_658), ShortDelivery)

    def test_growth_below_the_thresholds_is_silent(self):
        watch = DeliveryWatch()
        self.assertIsNone(watch.observe(60_000, 80_000))
        # +6,000 characters is churn, not a rewritten prompt.
        self.assertIsNone(watch.observe(66_000, 80_100))

    def test_one_warning_per_condition(self):
        watch = DeliveryWatch()
        watch.observe(*self.MEASURED_TRUNCATION[0])
        self.assertIsInstance(watch.observe(*self.MEASURED_TRUNCATION[1]), ShortDelivery)
        self.assertIsNone(watch.observe(300_000, 96_700))
        self.assertIsNone(watch.observe(400_000, 96_800))

    def test_shrinking_prompt_releases_the_latch(self):
        watch = DeliveryWatch()
        watch.observe(*self.MEASURED_TRUNCATION[0])
        self.assertIsInstance(watch.observe(*self.MEASURED_TRUNCATION[1]), ShortDelivery)
        # A chunked send starts smaller; the condition can be reported again.
        # 30,000 -> 80,000 characters with tokens tracking it is healthy ...
        self.assertIsNone(watch.observe(30_000, 60_000))
        self.assertIsNone(watch.observe(80_000, 140_000))
        # ... and 80,000 -> 200,000 characters on flat tokens is not.
        self.assertIsInstance(watch.observe(200_000, 140_500), ShortDelivery)

    def test_usage_reported_as_zero_is_ignored(self):
        watch = DeliveryWatch()
        self.assertIsNone(watch.observe(50_000, 0))
        self.assertIsNone(watch.observe(250_000, 0))

    def test_warning_is_logged_once(self):
        with self.assertLogs("delivery", level="WARNING") as captured:
            watch = DeliveryWatch()
            watch.observe(*self.MEASURED_TRUNCATION[0])
            watch.observe(*self.MEASURED_TRUNCATION[1])
            watch.observe(300_000, 96_700)
        self.assertEqual(len(captured.records), 1)

    def test_env_switch_disables_detection(self):
        with mock.patch.dict(os.environ, {DISABLE_ENV: "1"}):
            watch = DeliveryWatch()
            watch.observe(*self.MEASURED_TRUNCATION[0])
            self.assertIsNone(watch.observe(*self.MEASURED_TRUNCATION[1]))


if __name__ == "__main__":
    unittest.main()
