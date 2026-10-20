"""Exercise all pages and the real example-to-analysis Streamlit interaction."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, write_json


def main():
    from streamlit.testing.v1 import AppTest
    pages = ["app.py", "pages/1_Detect_Disease.py", "pages/2_Supported_Plants.py", "pages/3_Model_Performance.py", "pages/4_About.py"]
    checks = {}
    for page in pages:
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        if page != "app.py":
            app.switch_page(page).run(timeout=60)
        assert not app.exception, str(app.exception)
        checks[page] = "passed"
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
    app.switch_page("pages/1_Detect_Disease.py").run(timeout=60)
    assert not app.metric, "No assessment should appear before analysis."
    selector = next(box for box in app.selectbox if box.label == "Or try an example leaf")
    selector.select("Apple leaf").run(timeout=60)
    assert not app.exception and not app.metric
    next(button for button in app.button if button.label == "Analyse Leaf").click().run(timeout=120)
    assert not app.exception, str(app.exception)
    assert any(metric.label == "Leading confidence" for metric in app.metric)
    result = app.session_state["leaf_analysis"]["result"]
    assert len(result["top_three"]) == 3 and result["calibrated"]
    next(box for box in app.checkbox if box.label == "Show Grad-CAM explanation").check().run(timeout=120)
    assert not app.exception, str(app.exception)
    write_json(ROOT / "reports/app_verification.json", {"pages": checks, "no_prediction_before_analysis": True, "example_analysis": result, "gradcam_toggle_without_exception": True})
    print("All five pages and real example analysis/Grad-CAM interaction passed.")


if __name__ == "__main__":
    main()
