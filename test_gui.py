"""GUI smoke test: build the window, run a conversion, check the result lands.

The window is withdrawn, so this runs without a visible UI, but it exercises
the real code path: the worker thread, the progress queue, the polling loop and
the preview rendering.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ivc.engine import load_image
from ivc.gui import VectorConverterApp


def main() -> None:
    from test_pipeline import ensure_source

    app = VectorConverterApp()
    app.withdraw()

    app.source_image = load_image(ensure_source())
    app.source_path = os.path.abspath(ensure_source())
    app._redraw()
    app.start_convert()

    deadline = time.time() + 60
    while app.busy and time.time() < deadline:
        app.update()
        time.sleep(0.01)

    assert not app.busy, "conversion never finished"
    assert app.document is not None, "no document produced"
    assert app.document.path_count > 0, "document has no paths"
    assert not app.eps_btn.instate(["disabled"]), "EPS button still disabled"
    assert not app.svg_btn.instate(["disabled"]), "SVG button still disabled"
    assert "Vector ready" in app.status_var.get(), app.status_var.get()
    assert "paths" in app.stats_var.get(), app.stats_var.get()
    assert app.progress.cget("value") == 100

    app.view_var.set("vector")
    app._redraw()
    app.update()
    app.view_var.set("source")
    app._redraw()
    app.update()

    # a second run must be picked up from the queue just as cleanly
    app.vars["colors"].set(4)
    app.start_convert()
    deadline = time.time() + 60
    while app.busy and time.time() < deadline:
        app.update()
        time.sleep(0.01)
    assert not app.busy and app.document is not None

    # cancelling a run must not wedge the UI
    app.vars["upscale"].set(4.0)
    app.start_convert()
    app.cancel_flag.set()
    deadline = time.time() + 60
    while app.busy and time.time() < deadline:
        app.update()
        time.sleep(0.01)
    assert not app.busy, "cancel left the UI busy"
    assert str(app.convert_btn.cget("text")) == "Convert now"

    # the save path itself (no dialog, which would block here)
    from ivc.document import save_eps, save_svg
    out_eps = "testdata/gui_smoke.eps"
    save_eps(app.document, out_eps)
    save_svg(app.document, "testdata/gui_smoke.svg")
    assert os.path.getsize(out_eps) > 0

    app.destroy()
    print(f"OK - GUI flow works ({app.document.path_count} paths, "
          f"{os.path.getsize(out_eps) / 1024:.0f} KB EPS)")


if __name__ == "__main__":
    main()
