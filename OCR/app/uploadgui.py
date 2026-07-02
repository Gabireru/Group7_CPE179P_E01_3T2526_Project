# uploadgui.py
#
# Replaces camgui.py. Instead of a live camera feed, the user picks an
# image file from disk. The model runs and outputs:
#   1. The raw LaTeX string (selectable, copyable)
#   2. The LaTeX rendered as a math image via matplotlib
#
# Cross-compatible: Windows (laptop) and Raspberry Pi.
# No camera, no browser, no internet required at runtime.

import io
import base64
import threading

import flet as ft
from PIL import Image

# matplotlib is already in the environment (used in preprocess testing).
# We use it here purely to render LaTeX math to a PNG in memory.
try:
    import matplotlib
    matplotlib.use("Agg")           # non-interactive backend, safe on Pi
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ---------------------------------------------------------------------------
# LaTeX rendering helper
# ---------------------------------------------------------------------------

def render_latex_to_base64(latex_str: str, fontsize: int = 20) -> str | None:
    """
    Render a LaTeX string to a PNG image and return it as a base64 string
    for use in ft.Image(src_base64=...).

    Uses matplotlib's built-in math text renderer — no full LaTeX install
    needed, works on Pi with just matplotlib installed.

    Returns None if rendering fails (e.g. syntax error in the LaTeX).
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    try:
        # Wrap in $...$ for matplotlib math mode
        display_str = f"${latex_str}$"

        fig = plt.figure(figsize=(8, 1.5))
        fig.patch.set_facecolor("#0a1030")

        fig.text(
            0.5, 0.5,
            display_str,
            fontsize=fontsize,
            ha="center",
            va="center",
            color="white",
            usetex=False,       # use matplotlib's own math parser, not system LaTeX
        )

        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=120,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    except Exception:
        plt.close("all")
        return None


# ---------------------------------------------------------------------------
# Upload layout
# ---------------------------------------------------------------------------

def get_upload_layout(page: ft.Page, navigate_to, model_runner=None):
    dark_color = "#050a1f"
    bg_color = "#0a1030"

    # ── State ──────────────────────────────────────────────────────────────
    _state = {
        "pil_image": None,      # PIL image from the picked file
        "scanning": False,
    }

    # ── Controls ───────────────────────────────────────────────────────────

    preview_image = ft.Image(
        src="UI images/campg.jpg",      # placeholder until a file is picked
        width=480,
        height=300,
        fit=ft.BoxFit.CONTAIN,          # ft.ImageFit removed in flet 0.85+
        border_radius=12,
    )

    pick_label = ft.Text(
        "No file selected.",
        color=ft.Colors.WHITE60,
        size=11,
        italic=True,
    )

    status_text = ft.Text(
        "Pick an image to begin.",
        color=ft.Colors.WHITE60,
        size=11,
        italic=True,
    )

    result_label = ft.Text(
        "",
        color=ft.Colors.WHITE70,
        size=11,
        text_align=ft.TextAlign.CENTER,
    )

    # Raw LaTeX string — selectable so the user can copy manually too
    result_text = ft.Text(
        "",
        color=ft.Colors.WHITE,
        size=13,
        selectable=True,
        text_align=ft.TextAlign.CENTER,
        max_lines=4,
        overflow=ft.TextOverflow.ELLIPSIS,
    )

    # Rendered math image (matplotlib output).
    # src="" satisfies flet 0.85's required positional arg —
    # it gets replaced with src_base64 after inference runs.
    rendered_image = ft.Image(
        src="",
        visible=False,
        width=640,
        height=120,
        fit=ft.BoxFit.CONTAIN,
    )

    render_error = ft.Text(
        "",
        color=ft.Colors.ORANGE_300,
        size=11,
        italic=True,
        visible=False,
    )

    scan_btn = ft.ElevatedButton(
        content=ft.Text("SCAN", color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD),
        bgcolor=dark_color,
        disabled=True,
        style=ft.ButtonStyle(shape=ft.StadiumBorder()),
    )

    copy_btn = ft.ElevatedButton(
        content=ft.Text("COPY LaTeX", color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD),
        bgcolor=dark_color,
        disabled=True,
        style=ft.ButtonStyle(shape=ft.StadiumBorder()),
    )

    back_btn = ft.ElevatedButton(
        content=ft.Text("BACK", color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD),
        bgcolor=dark_color,
        style=ft.ButtonStyle(shape=ft.StadiumBorder()),
        on_click=lambda _: navigate_to("/main"),
    )

    # ── File picker ────────────────────────────────────────────────────────
    # In Flet 0.85, FilePicker.pick_files() is async and returns files
    # directly — no on_result callback needed. The button handler awaits
    # pick_files() and processes the result inline.

    file_picker = ft.FilePicker()
    page.services.append(file_picker)

    async def on_pick_clicked(e):
        files = await file_picker.pick_files(
            dialog_title="Select an equation image",
            allowed_extensions=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
            allow_multiple=False,
        )

        if not files:
            return

        picked = files[0]
        pick_label.value = picked.name

        try:
            pil = Image.open(picked.path).convert("RGB")
            _state["pil_image"] = pil

            # Show thumbnail preview
            thumb = pil.copy()
            thumb.thumbnail((480, 300))
            buf = io.BytesIO()
            thumb.save(buf, format="PNG")
            preview_image.src = None
            preview_image.src_base64 = base64.b64encode(buf.getvalue()).decode()

            status_text.value = "Image loaded. Press SCAN."
            scan_btn.disabled = False

            # Clear previous results
            result_label.value = ""
            result_text.value = ""
            rendered_image.visible = False
            render_error.visible = False
            copy_btn.disabled = True

        except Exception as exc:
            status_text.value = f"Could not open image: {exc}"
            scan_btn.disabled = True

        page.update()

    pick_btn = ft.ElevatedButton(
        content=ft.Text("PICK IMAGE", color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD),
        bgcolor=dark_color,
        style=ft.ButtonStyle(shape=ft.StadiumBorder()),
        on_click=on_pick_clicked,
    )

    # ── Inference ──────────────────────────────────────────────────────────

    def _do_inference():
        """Runs in a worker thread — never call page.update() from here directly,
        just mutate controls then call page.update() at the end."""
        try:
            if model_runner is None:
                raise RuntimeError("No model runner provided.")
            if not model_runner.ready:
                raise RuntimeError(
                    model_runner.error or "Model is still loading — please wait a moment."
                )
            if _state["pil_image"] is None:
                raise RuntimeError("No image loaded.")

            latex = model_runner.predict(_state["pil_image"])

            # Store for the COPY button on the main page too
            page.session.store.set("last_latex", latex)

            result_label.value = "Predicted LaTeX (raw):"
            result_text.value = latex
            copy_btn.disabled = False
            status_text.value = "Done."

            # Attempt to render the LaTeX as a math image
            b64 = render_latex_to_base64(latex)
            if b64:
                rendered_image.src_base64 = b64
                rendered_image.src = None
                rendered_image.visible = True
                render_error.visible = False
            else:
                rendered_image.visible = False
                render_error.value = (
                    "Could not render as math — formula may contain unsupported LaTeX commands. "
                    "The raw string above is still correct."
                )
                render_error.visible = True

        except Exception as exc:
            result_label.value = "Error:"
            result_text.value = str(exc)
            rendered_image.visible = False
            status_text.value = ""
            copy_btn.disabled = True

        finally:
            _state["scanning"] = False
            scan_btn.disabled = False
            scan_btn.content.value = "SCAN"
            page.update()

    def on_scan(e):
        if _state["scanning"] or _state["pil_image"] is None:
            return
        _state["scanning"] = True
        scan_btn.disabled = True
        scan_btn.content.value = "SCANNING…"
        result_label.value = ""
        result_text.value = ""
        rendered_image.visible = False
        render_error.visible = False
        status_text.value = "Running model…"
        copy_btn.disabled = True
        page.update()
        page.run_thread(_do_inference)

    scan_btn.on_click = on_scan

    # ── Copy ───────────────────────────────────────────────────────────────

    def on_copy(e):
        latex = page.session.store.get("last_latex")
        if latex:
            page.set_clipboard(latex)
            snack = ft.SnackBar(
                content=ft.Text("LaTeX copied to clipboard!", color=ft.Colors.WHITE),
                bgcolor="#1a2a6c",
                duration=2000,
            )
        else:
            snack = ft.SnackBar(
                content=ft.Text("No result yet — run SCAN first.", color=ft.Colors.WHITE),
                bgcolor="#6c1a1a",
                duration=2000,
            )
        page.overlay.append(snack)
        snack.open = True
        page.update()

    copy_btn.on_click = on_copy

    # ── Layout ─────────────────────────────────────────────────────────────

    return ft.Column(
        controls=[
            # ── Top bar ──
            ft.Container(
                content=ft.Row(
                    controls=[
                        back_btn,
                        ft.Text(
                            "Formula Scanner — Upload",
                            color=ft.Colors.WHITE,
                            size=16,
                            weight=ft.FontWeight.BOLD,
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
                padding=16,
                bgcolor=dark_color,
            ),

            # ── Image preview ──
            ft.Container(
                content=ft.Column(
                    controls=[
                        preview_image,
                        pick_label,
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=6,
                ),
                alignment=ft.Alignment.CENTER,
                expand=True,
                bgcolor=bg_color,
                padding=16,
            ),

            # ── Result area ──
            ft.Container(
                content=ft.Column(
                    controls=[
                        status_text,
                        result_label,
                        result_text,
                        rendered_image,
                        render_error,
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=4,
                ),
                padding=ft.Padding(left=20, top=8, right=20, bottom=8),
                bgcolor=bg_color,
            ),

            # ── Action bar ──
            ft.Container(
                content=ft.Row(
                    controls=[pick_btn, scan_btn, copy_btn],
                    alignment=ft.MainAxisAlignment.CENTER,
                    spacing=12,
                ),
                padding=16,
                bgcolor=dark_color,
            ),
        ],
        spacing=0,
        expand=True,
    )