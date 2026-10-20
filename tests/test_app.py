import pytest


@pytest.mark.parametrize("file", ["app.py", "pages/1_Detect_Disease.py", "pages/2_Supported_Plants.py", "pages/3_Model_Performance.py", "pages/4_About.py"])
def test_pages_start_without_artifacts(file):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file("app.py").run(timeout=30)
    if file != "app.py":
        app.switch_page(file).run(timeout=30)
    assert not app.exception
