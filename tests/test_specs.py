from rx590gme import RX590_GME, spec_sheet


def test_shader_array_matches_the_real_card():
    assert RX590_GME.compute_units == 36
    assert RX590_GME.stream_processors == 2304
    assert RX590_GME.tmus == 144
    assert RX590_GME.rops == 32


def test_derived_throughput():
    assert RX590_GME.memory_bandwidth_gbs == 256
    assert round(RX590_GME.fp32_gflops()) == 6359
    assert round(RX590_GME.fp32_gflops(RX590_GME.base_clock_mhz)) == 5792
    assert round(RX590_GME.pixel_fillrate_gpix(), 1) == 44.2
    assert round(RX590_GME.texture_rate_gtex(), 1) == 198.7


def test_spec_sheet_mentions_key_facts():
    sheet = spec_sheet()
    for text in ("Polaris 20 XTX", "2304", "8 GB GDDR5", "256-bit", "175 W"):
        assert text in sheet
