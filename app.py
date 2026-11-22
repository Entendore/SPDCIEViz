import sys
import math
import numpy as np
from dataclasses import dataclass
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox, QColorDialog, QComboBox, QSlider,
    QSizePolicy, QFormLayout, QFileDialog, QFrame, QTabWidget, 
    QListWidget, QTextBrowser, QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ==============================================================================
# SECTION 1 — SPECTRAL DOMAIN DEFINITIONS
# ==============================================================================
SPECTRUM_MIN_HZ = 400e12      # 750 nm (red)
SPECTRUM_MAX_HZ = 790e12      # 380 nm (violet)
N_SAMPLES       = 400
FREQS_HZ        = np.linspace(SPECTRUM_MIN_HZ, SPECTRUM_MAX_HZ, N_SAMPLES)
FREQS_THZ       = FREQS_HZ / 1e12
WAVELENGTHS_NM  = 3e8 / FREQS_HZ * 1e9

H_PLANCK     = 6.62607015e-34
K_BOLTZMANN  = 1.380649e-23
C_LIGHT      = 299_792_458.0

D65_X = 0.3127
D65_Y = 0.3290

# ==============================================================================
# SECTION 2 — CIE 1931 2° STANDARD OBSERVER DATA
# ==============================================================================
CIE_1931_2DEG_DATA = {
    380: (0.001368, 0.000039, 0.006450), 385: (0.002236, 0.000064, 0.010550),
    390: (0.004243, 0.000120, 0.020050), 395: (0.007650, 0.000217, 0.036210),
    400: (0.014310, 0.000396, 0.067850), 405: (0.023190, 0.000640, 0.110200),
    410: (0.043510, 0.001210, 0.207400), 415: (0.077630, 0.002180, 0.371300),
    420: (0.134380, 0.004000, 0.645600), 425: (0.214770, 0.007300, 1.039050),
    430: (0.283900, 0.011600, 1.385600), 435: (0.328500, 0.016840, 1.622960),
    440: (0.348280, 0.023000, 1.747060), 445: (0.348060, 0.029800, 1.782600),
    450: (0.336200, 0.038000, 1.772110), 455: (0.318700, 0.048000, 1.744100),
    460: (0.290800, 0.060000, 1.669200), 465: (0.251100, 0.073900, 1.528100),
    470: (0.195360, 0.090980, 1.287640), 475: (0.142100, 0.112600, 1.041900),
    480: (0.095640, 0.139020, 0.812950), 485: (0.057950, 0.169300, 0.616200),
    490: (0.032010, 0.208020, 0.465180), 495: (0.014700, 0.258600, 0.353300),
    500: (0.004900, 0.323000, 0.272000), 505: (0.002400, 0.407300, 0.212300),
    510: (0.009300, 0.503000, 0.158200), 515: (0.029100, 0.608200, 0.111700),
    520: (0.063270, 0.710000, 0.078250), 525: (0.109600, 0.793200, 0.057250),
    530: (0.165500, 0.862000, 0.042160), 535: (0.225750, 0.914850, 0.029840),
    540: (0.290400, 0.954000, 0.020300), 545: (0.359700, 0.980300, 0.013400),
    550: (0.433450, 0.994950, 0.008750), 555: (0.512050, 1.000000, 0.005750),
    560: (0.594500, 0.995000, 0.003900), 565: (0.678400, 0.978600, 0.002750),
    570: (0.762100, 0.952000, 0.002100), 575: (0.842500, 0.915400, 0.001800),
    580: (0.916300, 0.870000, 0.001650), 585: (0.978600, 0.816300, 0.001400),
    590: (1.026300, 0.757000, 0.001100), 595: (1.056700, 0.694900, 0.001000),
    600: (1.062200, 0.631000, 0.000800), 605: (1.045600, 0.566800, 0.000600),
    610: (1.002600, 0.503000, 0.000340), 615: (0.938400, 0.441200, 0.000240),
    620: (0.854450, 0.381000, 0.000190), 625: (0.751400, 0.321000, 0.000100),
    630: (0.642400, 0.265000, 0.000050), 635: (0.541900, 0.217000, 0.000030),
    640: (0.447900, 0.175000, 0.000020), 645: (0.360800, 0.138200, 0.000010),
    650: (0.283500, 0.107000, 0.000000), 655: (0.218700, 0.081600, 0.000000),
    660: (0.164900, 0.061000, 0.000000), 665: (0.121200, 0.044580, 0.000000),
    670: (0.087400, 0.032000, 0.000000), 675: (0.063600, 0.023200, 0.000000),
    680: (0.046770, 0.017000, 0.000000), 685: (0.032900, 0.011920, 0.000000),
    690: (0.022700, 0.008210, 0.000000), 695: (0.015840, 0.005723, 0.000000),
    700: (0.011359, 0.004102, 0.000000), 710: (0.005790, 0.002091, 0.000000),
    720: (0.002899, 0.001047, 0.000000), 730: (0.001440, 0.000520, 0.000000),
    740: (0.000690, 0.000249, 0.000000), 750: (0.000332, 0.000120, 0.000000),
}

def interpolate_cie_cmf(wavelengths_nm: np.ndarray) -> tuple:
    wl_cie = np.array(sorted(CIE_1931_2DEG_DATA.keys()))
    x_cie = np.array([CIE_1931_2DEG_DATA[wl][0] for wl in wl_cie])
    y_cie = np.array([CIE_1931_2DEG_DATA[wl][1] for wl in wl_cie])
    z_cie = np.array([CIE_1931_2DEG_DATA[wl][2] for wl in wl_cie])
    return (np.interp(wavelengths_nm, wl_cie, x_cie, left=0, right=0),
            np.interp(wavelengths_nm, wl_cie, y_cie, left=0, right=0),
            np.interp(wavelengths_nm, wl_cie, z_cie, left=0, right=0))

X_CMF, Y_CMF, Z_CMF = interpolate_cie_cmf(WAVELENGTHS_NM)
SCALE_Y = 1.0 / np.trapz(Y_CMF, FREQS_HZ)

_locus_X = X_CMF / (X_CMF + Y_CMF + Z_CMF + 1e-12)
_locus_Y = Y_CMF / (X_CMF + Y_CMF + Z_CMF + 1e-12)
LOCUS_POINTS = np.array([_locus_X, _locus_Y]).T

PRESETS = [
    "Custom RGB", 
    "D65 Daylight", 
    "Blackbody 5000K (Horizon)", 
    "Blackbody 3000K (Tungsten)", 
    "Red Laser (633nm)", 
    "Green Laser (532nm)",
    "Blue Laser (445nm)",
    "Sodium Lamp (589nm)",
    "Cyan Filter (490nm)", 
    "Magenta Notch Filter"
]

# ==============================================================================
# SECTION 3 — SPECTRAL PHYSICS
# ==============================================================================
class SpectralPhysics:

    @staticmethod
    def srgb_to_linear(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    @staticmethod
    def linear_to_srgb(c: float) -> float:
        return 12.92 * c if c <= 0.0031308 else 1.055 * (max(c, 0.0) ** (1 / 2.4)) - 0.055

    @classmethod
    def wavelength_to_rgb(cls, wl_nm: float) -> tuple:
        freq = 3e8 / (wl_nm * 1e-9)
        spd = np.exp(-((FREQS_HZ - freq)**2) / (2 * (1e12)**2))
        rgb, _ = cls.spd_to_rgb(spd)
        return rgb

    @classmethod
    def rgb_to_spd(cls, rgb: tuple) -> np.ndarray:
        r = cls.srgb_to_linear(max(0.0, rgb[0]))
        g = cls.srgb_to_linear(max(0.0, rgb[1]))
        b = cls.srgb_to_linear(max(0.0, rgb[2]))
        cmin = min(r, g, b)

        _WHITE_SPD = np.ones_like(FREQS_HZ)
        _RED_SPD   = np.exp(-((FREQS_HZ - 430e12) ** 2) / (2 * (35e12) ** 2))
        _GREEN_SPD = np.exp(-((FREQS_HZ - 545e12) ** 2) / (2 * (35e12) ** 2))
        _BLUE_SPD  = np.exp(-((FREQS_HZ - 680e12) ** 2) / (2 * (35e12) ** 2))

        spd = cmin * _WHITE_SPD
        if r > cmin: spd = spd + (r - cmin) * _RED_SPD
        if g > cmin: spd = spd + (g - cmin) * _GREEN_SPD
        if b > cmin: spd = spd + (b - cmin) * _BLUE_SPD

        m = spd.max()
        return spd / m if m > 1e-12 else np.zeros_like(spd)

    @staticmethod
    def get_preset_spd(name: str) -> np.ndarray:
        if name == "D65 Daylight":
            return np.ones_like(FREQS_HZ) * 1.0
        elif name.startswith("Blackbody"):
            T = 5000 if "5000K" in name else 3000
            wl_m = WAVELENGTHS_NM * 1e-9
            rad = (2 * H_PLANCK * C_LIGHT**2 / wl_m**5) / (np.exp(H_PLANCK * C_LIGHT / (wl_m * K_BOLTZMANN * T)) - 1)
            return rad / rad.max()
        elif name == "Red Laser (633nm)":
            return np.exp(-((FREQS_HZ - 474e12) ** 2) / (2 * (2e12) ** 2))
        elif name == "Green Laser (532nm)":
            return np.exp(-((FREQS_HZ - 564e12) ** 2) / (2 * (2e12) ** 2))
        elif name == "Blue Laser (445nm)":
            return np.exp(-((FREQS_HZ - 673e12) ** 2) / (2 * (2e12) ** 2))
        elif name == "Sodium Lamp (589nm)":
            return np.exp(-((FREQS_HZ - 509e12) ** 2) / (2 * (5e12) ** 2))
        elif name == "Cyan Filter (490nm)":
            return np.exp(-((FREQS_HZ - 612e12) ** 2) / (2 * (30e12) ** 2))
        elif name == "Magenta Notch Filter":
            return 1.0 - np.exp(-((FREQS_HZ - 545e12) ** 2) / (2 * (40e12) ** 2))
        return np.zeros_like(FREQS_HZ)

    @staticmethod
    def additive_mix(spd_a: np.ndarray, spd_b: np.ndarray) -> np.ndarray:
        return spd_a + spd_b

    @staticmethod
    def subtractive_mix(t_a: np.ndarray, t_b: np.ndarray) -> np.ndarray:
        return np.clip(t_a, 0, 1) * np.clip(t_b, 0, 1)

    @staticmethod
    def beer_lambert_transmittance(spd: np.ndarray, thickness: float, k_max: float = 4.0) -> np.ndarray:
        absorption = (1.0 - np.clip(spd, 0.0, 1.0)) * k_max
        return np.exp(-absorption * thickness)

    @classmethod
    def spd_to_xyz(cls, spd: np.ndarray, normalize_y: bool = False) -> tuple:
        X = float(np.trapz(spd * X_CMF, FREQS_HZ) * SCALE_Y)
        Y = float(np.trapz(spd * Y_CMF, FREQS_HZ) * SCALE_Y)
        Z = float(np.trapz(spd * Z_CMF, FREQS_HZ) * SCALE_Y)
        if normalize_y and Y > 1e-12:
            return (X / Y, 1.0, Z / Y)
        return (X, Y, Z)

    @classmethod
    def spd_to_rgb(cls, spd: np.ndarray) -> tuple:
        X, Y, Z = cls.spd_to_xyz(spd, normalize_y=True)
        total = X + Y + Z
        if total < 1e-12: return (0.0, 0.0, 0.0, False)
        
        x, y, z = X / total, Y / total, Z / total
        r = x * 3.2406 + y * -1.5372 + z * -0.4986
        g = x * -0.9689 + y * 1.8758 + z * 0.0415
        b = x * 0.0557 + y * -0.2040 + z * 1.0570

        is_in_gamut = (0.0 <= r <= 1.0) and (0.0 <= g <= 1.0) and (0.0 <= b <= 1.0)
        r = cls.linear_to_srgb(r)
        g = cls.linear_to_srgb(g)
        b = cls.linear_to_srgb(b)
        
        return (max(0.0, min(1.0, r)), max(0.0, min(1.0, g)), max(0.0, min(1.0, b)), is_in_gamut)

    @staticmethod
    def xyz_to_lab(X: float, Y: float, Z: float) -> tuple:
        Xn, Yn, Zn = 0.95047, 1.0, 1.08883
        def f(t):
            return t**(1/3) if t > (6/29)**3 else (1/3) * (29/6)**2 * t + 4/29
        fx, fy, fz = f(X / Xn), f(Y / Yn), f(Z / Zn)
        return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))

    @staticmethod
    def calculate_cct(X: float, Y: float, Z: float) -> float:
        total = X + Y + Z
        if total < 1e-12: return float('nan')
        x, y = X / total, Y / total
        n = (x - 0.3320) / (0.1858 - y)
        cct = 449 * n**3 + 3525 * n**2 + 6823.3 * n + 5520.33
        return round(cct) if 1000 < cct < 40000 else float('nan')

    @staticmethod
    def delta_e_76(lab1: tuple, lab2: tuple) -> float:
        return math.sqrt((lab1[0]-lab2[0])**2 + (lab1[1]-lab2[1])**2 + (lab1[2]-lab2[2])**2)

# Precompute Planckian Locus (1500K - 10000K)
PLANCKIAN_LOCUS = []
for T in range(1500, 10001, 250):
    wl_m = WAVELENGTHS_NM * 1e-9
    rad = (2 * H_PLANCK * C_LIGHT**2 / wl_m**5) / (np.exp(H_PLANCK * C_LIGHT / (wl_m * K_BOLTZMANN * T)) - 1)
    X, Y, Z = SpectralPhysics.spd_to_xyz(rad, normalize_y=False)
    tot = X + Y + Z + 1e-12
    PLANCKIAN_LOCUS.append((T, X/tot, Y/tot))


# ==============================================================================
# SECTION 4 — SHARED UI COMPONENTS
# ==============================================================================
DARK_QSS = """
    QMainWindow, QWidget { background-color: #1a1d24; color: #e0e0e0; font-size: 10pt; }
    QGroupBox {
        border: 1px solid #3c3f4a;
        border-radius: 6px;
        margin-top: 14px;
        padding: 12px;
        font-weight: bold;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
        color: #8ad8ff;
    }
    QPushButton {
        background-color: #2a2e38;
        border: 1px solid #4a4f5a;
        border-radius: 4px;
        padding: 6px;
    }
    QPushButton:hover { background-color: #353a45; border-color: #8ad8ff; }
    QPushButton:disabled { background-color: #1a1d24; color: #555; border-color: #2a2e38; }
    QComboBox {
        background-color: #2a2e38;
        border: 1px solid #4a4f5a;
        border-radius: 4px;
        padding: 4px;
    }
    QComboBox::drop-down { border: none; }
    QComboBox QAbstractItemView {
        background-color: #2a2e38;
        selection-background-color: #8ad8ff;
        selection-color: #1a1d24;
    }
    QSlider::groove:horizontal {
        height: 4px;
        background: #4a4f5a;
        border-radius: 2px;
    }
    QSlider::handle:horizontal {
        background: #8ad8ff;
        width: 14px;
        margin: -5px 0;
        border-radius: 7px;
    }
    QListWidget {
        background-color: #11141a;
        border: 1px solid #3c3f4a;
        border-radius: 4px;
    }
    QListWidget::item:selected {
        background-color: #8ad8ff;
        color: #1a1d24;
        font-weight: bold;
    }
    QTextBrowser {
        background-color: #11141a;
        border: 1px solid #3c3f4a;
        border-radius: 4px;
    }
    QTabWidget::pane { border: 1px solid #3c3f4a; border-radius: 4px; }
    QTabBar::tab {
        background: #1a1d24;
        padding: 8px 16px;
        border: 1px solid #3c3f4a;
        border-bottom: none;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
    }
    QTabBar::tab:selected { background: #2a2e38; color: #8ad8ff; }
"""

def make_swatch(title: str) -> QFrame:
    g = QFrame()
    g.setStyleSheet("background: transparent;")
    v = QVBoxLayout(g)
    v.setContentsMargins(0, 0, 0, 0)
    v.setAlignment(Qt.AlignCenter)
    
    lbl_title = QLabel(title)
    lbl_title.setAlignment(Qt.AlignCenter)
    lbl_title.setStyleSheet("color: #8ad8ff; font-weight: bold;")
    
    swatch = QLabel()
    swatch.setFixedSize(160, 60)
    swatch.setAlignment(Qt.AlignCenter)
    swatch.setStyleSheet("background:#000; border:1px solid #888; color:#fff; font-weight:bold;")
    
    info = QLabel("—")
    info.setAlignment(Qt.AlignCenter)
    info.setStyleSheet("color:#aaa; font-family: monospace;")
    
    v.addWidget(lbl_title)
    v.addWidget(swatch)
    v.addWidget(info)
    g._swatch = swatch
    g._info   = info
    return g

def set_swatch_color(frame: QFrame, rgb: tuple, label: str) -> None:
    r = max(0, min(255, int(round(rgb[0] * 255))))
    g = max(0, min(255, int(round(rgb[1] * 255))))
    b = max(0, min(255, int(round(rgb[2] * 255))))
    sw = frame._swatch
    info = frame._info
    sw.setStyleSheet(f"background:rgb({r},{g},{b}); border:1px solid #888; color:#fff; font-weight:bold;")
    info.setText(label)


# ==============================================================================
# SECTION 5 — TAB 1: COLORIMETRY LAB
# ==============================================================================
@dataclass(slots=True)
class MixState:
    color_a: QColor
    preset_a: str
    color_b: QColor
    preset_b: str
    additive: bool
    thickness: float

class ColorimetryLab(QWidget):
    def __init__(self):
        super().__init__()
        self.state = MixState(
            color_a=QColor("#FF3030"), preset_a="Custom RGB",
            color_b=QColor("#3060FF"), preset_b="Custom RGB",
            additive=True, thickness=1.0,
        )

        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        left = QVBoxLayout()
        right = QVBoxLayout()
        root.addLayout(left, 1)
        root.addLayout(right, 2)

        self._build_controls(left)
        self._build_visuals(right)
        self.update_render()

    def _build_controls(self, L: QVBoxLayout) -> None:
        # --- CHANNEL A ---
        grp_a = QGroupBox("Channel A")
        va = QVBoxLayout(grp_a)
        self.combo_a = QComboBox()
        self.combo_a.addItems(PRESETS)
        self.combo_a.currentTextChanged.connect(lambda txt: self._on_preset_change('A', txt))
        
        self.btn_a = QPushButton("Select Custom Color A")
        self.btn_a.clicked.connect(lambda: self._pick_color("A"))
        
        self.swatch_a = make_swatch("Channel A")
        va.addWidget(self.combo_a)
        va.addWidget(self.btn_a)
        va.addWidget(self.swatch_a)
        L.addWidget(grp_a)

        # --- SWAP BUTTON ---
        btn_swap = QPushButton("⇅ Swap A ↔ B")
        btn_swap.clicked.connect(self._swap_channels)
        L.addWidget(btn_swap)

        # --- CHANNEL B ---
        grp_b = QGroupBox("Channel B")
        vb = QVBoxLayout(grp_b)
        self.combo_b = QComboBox()
        self.combo_b.addItems(PRESETS)
        self.combo_b.currentTextChanged.connect(lambda txt: self._on_preset_change('B', txt))
        
        self.btn_b = QPushButton("Select Custom Color B")
        self.btn_b.clicked.connect(lambda: self._pick_color("B"))
        
        self.swatch_b = make_swatch("Channel B")
        vb.addWidget(self.combo_b)
        vb.addWidget(self.btn_b)
        vb.addWidget(self.swatch_b)
        L.addWidget(grp_b)

        # --- MIX PHYSICS ---
        thick = QGroupBox("Mix Physics & Transparency")
        tv = QVBoxLayout(thick)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Additive (Light + Light)", "Subtractive (Pigment + Pigment)"])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        
        self.thick_label = QLabel("Pigment Thickness: 1.0")
        self.thick_slider = QSlider(Qt.Horizontal)
        self.thick_slider.setRange(1, 100)
        self.thick_slider.setValue(10)
        self.thick_slider.valueChanged.connect(self._on_thickness)
        
        tv.addWidget(self.mode_combo)
        tv.addWidget(self.thick_label)
        tv.addWidget(self.thick_slider)
        L.addWidget(thick)

        # --- MIXED OUTPUT & DATA ---
        out_grp = QGroupBox("Mixed Output & Advanced Colorimetry")
        ov = QVBoxLayout(out_grp)
        
        self.swatch_m = make_swatch("Mixed Output")
        ov.addWidget(self.swatch_m)
        
        data_form = QFormLayout()
        self.lbl_xyz = QLabel("X: 0.0  |  Y: 0.0  |  Z: 0.0")
        self.lbl_xyy = QLabel("x: 0.0  |  y: 0.0  |  Y: 0.0")
        self.lbl_lab = QLabel("L*: 0.0  |  a*: 0.0  |  b*: 0.0")
        self.lbl_cct = QLabel("CCT: — K")
        self.lbl_de = QLabel("ΔE (A vs Mix): 0.0")
        self.lbl_gamut = QLabel("Gamut: Valid")
        self.lbl_gamut.setStyleSheet("color: #4CAF50; font-weight: bold;")
        
        for lbl in [self.lbl_xyz, self.lbl_xyy, self.lbl_lab, self.lbl_cct, self.lbl_de]:
            lbl.setStyleSheet("font-family: monospace; color: #ccc;")
            data_form.addRow(lbl)
        data_form.addRow(self.lbl_gamut)
        ov.addLayout(data_form)

        btn_export = QPushButton("Export Mixed SPD to CSV")
        btn_export.clicked.connect(self._export_csv)
        ov.addWidget(btn_export)
        
        L.addWidget(out_grp)
        L.addStretch(1)

    def _build_visuals(self, L: QVBoxLayout) -> None:
        self.figure = Figure(figsize=(8, 6), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        L.addWidget(self.canvas, 3)

    def _pick_color(self, target: str) -> None:
        cur = self.state.color_a if target == "A" else self.state.color_b
        c = QColorDialog.getColor(cur, self, f"Select Color {target}")
        if c.isValid():
            if target == "A": self.state.color_a = c
            else:             self.state.color_b = c
            self._refresh_button_colors()
            self.update_render()

    def _on_preset_change(self, target: str, text: str) -> None:
        if target == 'A':
            self.state.preset_a = text
            self.btn_a.setEnabled(text == "Custom RGB")
        else:
            self.state.preset_b = text
            self.btn_b.setEnabled(text == "Custom RGB")
        self.update_render()

    def _on_mode_changed(self, text: str) -> None:
        self.state.additive = text.startswith("Additive")
        self.update_render()

    def _on_thickness(self, v: int) -> None:
        self.state.thickness = v / 10.0
        self.thick_label.setText(f"Pigment Thickness: {self.state.thickness:.1f}")
        self.update_render()

    def _swap_channels(self) -> None:
        self.state.color_a, self.state.color_b = self.state.color_b, self.state.color_a
        self.state.preset_a, self.state.preset_b = self.state.preset_b, self.state.preset_a
        self.combo_a.setCurrentText(self.state.preset_a)
        self.combo_b.setCurrentText(self.state.preset_b)
        self._refresh_button_colors()
        self.update_render()

    def _export_csv(self) -> None:
        if not hasattr(self, 'latest_mixed_spd'): return
        path, _ = QFileDialog.getSaveFileName(self, "Save Mixed SPD", "mixed_spd.csv", "CSV Files (*.csv)")
        if path:
            data = np.column_stack((WAVELENGTHS_NM, FREQS_THZ, self.latest_mixed_spd))
            np.savetxt(path, data, delimiter=',', header='Wavelength(nm),Frequency(THz),Normalized_Intensity', comments='')
            QMessageBox.information(self, "Export Successful", f"Exported SPD to {path}")

    def _refresh_button_colors(self) -> None:
        c = self.state.color_a
        self.btn_a.setStyleSheet(f"background-color:{c.name()}; color:{self._contrast(c)}; font-weight:bold; padding:8px;")
        c = self.state.color_b
        self.btn_b.setStyleSheet(f"background-color:{c.name()}; color:{self._contrast(c)}; font-weight:bold; padding:8px;")

    @staticmethod
    def _contrast(c: QColor) -> str:
        lum = 0.2126 * c.redF() + 0.7152 * c.greenF() + 0.0722 * c.blueF()
        return "#000" if lum > 0.55 else "#FFF"

    def _resolve_spd(self, target: str) -> np.ndarray:
        if target == 'A':
            if self.state.preset_a != "Custom RGB": return SpectralPhysics.get_preset_spd(self.state.preset_a)
            rgb = (self.state.color_a.redF(), self.state.color_a.greenF(), self.state.color_a.blueF())
            return SpectralPhysics.rgb_to_spd(rgb)
        else:
            if self.state.preset_b != "Custom RGB": return SpectralPhysics.get_preset_spd(self.state.preset_b)
            rgb = (self.state.color_b.redF(), self.state.color_b.greenF(), self.state.color_b.blueF())
            return SpectralPhysics.rgb_to_spd(rgb)

    def update_render(self) -> None:
        st = self.state

        spd_a = self._resolve_spd('A')
        spd_b = self._resolve_spd('B')

        if st.additive:
            mixed = SpectralPhysics.additive_mix(spd_a, spd_b)
        else:
            t_a = SpectralPhysics.beer_lambert_transmittance(spd_a, st.thickness)
            t_b = SpectralPhysics.beer_lambert_transmittance(spd_b, st.thickness)
            mixed = SpectralPhysics.subtractive_mix(t_a, t_b)

        m = mixed.max()
        mixed_norm = mixed / m if m > 1e-12 else mixed
        self.latest_mixed_spd = mixed_norm

        rgb_a_disp, _ = SpectralPhysics.spd_to_rgb(spd_a)
        set_swatch_color(self.swatch_a, rgb_a_disp, f"A: {st.preset_a}")
        
        rgb_b_disp, _ = SpectralPhysics.spd_to_rgb(spd_b)
        set_swatch_color(self.swatch_b, rgb_b_disp, f"B: {st.preset_b}")

        mixed_rgb, is_in_gamut = SpectralPhysics.spd_to_rgb(mixed_norm)
        
        X_abs, Y_abs, Z_abs = SpectralPhysics.spd_to_xyz(mixed_norm, normalize_y=False)
        if Y_abs > 1e-12:
            L_val, a_val, b_val = SpectralPhysics.xyz_to_lab(X_abs/Y_abs, 1.0, Z_abs/Y_abs)
        else:
            L_val, a_val, b_val = 0.0, 0.0, 0.0
            
        x_coord = X_abs / (X_abs + Y_abs + Z_abs + 1e-12)
        y_coord = Y_abs / (X_abs + Y_abs + Z_abs + 1e-12)
        
        self.lbl_xyz.setText(f"X: {X_abs:.3f}  |  Y: {Y_abs:.3f}  |  Z: {Z_abs:.3f}")
        self.lbl_xyy.setText(f"x: {x_coord:.3f}  |  y: {y_coord:.3f}  |  Y: {Y_abs:.3f}")
        self.lbl_lab.setText(f"L*: {L_val:.1f}  |  a*: {a_val:.1f}  |  b*: {b_val:.1f}")
        
        cct = SpectralPhysics.calculate_cct(X_abs, Y_abs, Z_abs)
        self.lbl_cct.setText(f"CCT: {int(cct)} K" if not np.isnan(cct) else "CCT: — K")
        
        X_a, Y_a, Z_a = SpectralPhysics.spd_to_xyz(spd_a, normalize_y=False)
        if Y_a > 1e-12 and Y_abs > 1e-12:
            lab_a = SpectralPhysics.xyz_to_lab(X_a/Y_a, 1.0, Z_a/Y_a)
            dE = SpectralPhysics.delta_e_76(lab_a, (L_val, a_val, b_val))
            self.lbl_de.setText(f"ΔE (A vs Mix): {dE:.1f}")
        else:
            self.lbl_de.setText("ΔE (A vs Mix): —")
        
        if is_in_gamut:
            self.lbl_gamut.setText("Gamut: Valid (Inside sRGB)")
            self.lbl_gamut.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.lbl_gamut.setText("Gamut: WARNING (Clamped)")
            self.lbl_gamut.setStyleSheet("color: #FF5252; font-weight: bold;")

        h = "#{:02X}{:02X}{:02X}".format(int(mixed_rgb[0]*255), int(mixed_rgb[1]*255), int(mixed_rgb[2]*255))
        set_swatch_color(self.swatch_m, mixed_rgb, f"Mixed: {h}")

        self.figure.clear()
        gs = self.figure.add_gridspec(2, 1, height_ratios=[2, 1.2])
        
        ax1 = self.figure.add_subplot(gs[0])
        ax1.set_facecolor("#11141a")
        ax1.set_title("Spectral Power Distributions & Luminance Footprint", color="#eee")
        ax1.set_xlabel("Wavelength (nm)", color="#eee")
        ax1.set_ylabel("Normalized intensity", color="#eee")
        ax1.tick_params(colors="#bbb")
        for s in ax1.spines.values(): s.set_color("#555")
        ax1.grid(True, color="#333", alpha=0.6)

        ax1_top = ax1.secondary_xaxis("top",
            functions=(lambda x: 3e8 / (x * 1e-9) / 1e12, lambda x: 3e8 / (x * 1e12) * 1e9))
        ax1_top.set_xlabel("Frequency (THz)", color="#bbb")
        ax1_top.tick_params(colors="#bbb")

        gradient = np.zeros((100, N_SAMPLES, 3))
        for i, wl in enumerate(WAVELENGTHS_NM):
            gradient[:, i] = SpectralPhysics.wavelength_to_rgb(wl)
        ax1.imshow(gradient, extent=[WAVELENGTHS_NM.min(), WAVELENGTHS_NM.max(), 0, 1.05], 
                   aspect='auto', zorder=0, alpha=0.15, origin='lower')

        ax1.plot(WAVELENGTHS_NM, spd_a, color=self.state.color_a.name(), alpha=0.8, lw=2, label="A (input SPD)")
        ax1.plot(WAVELENGTHS_NM, spd_b, color=self.state.color_b.name(), alpha=0.8, lw=2, label="B (input SPD)")
        ax1.plot(WAVELENGTHS_NM, mixed_norm, color="#FFFFFF", lw=2.4, label="Mixed SPD", zorder=3)
        ax1.fill_between(WAVELENGTHS_NM, 0, mixed_norm, color="#FFFFFF", alpha=0.1, zorder=2)

        luminance_footprint = (mixed_norm * Y_CMF) / Y_CMF.max()
        ax1.fill_between(WAVELENGTHS_NM, 0, luminance_footprint, color='#FFD700', alpha=0.35, label="Perceptual Luminance", zorder=2)
        
        ax1.set_xlim(WAVELENGTHS_NM.max(), WAVELENGTHS_NM.min())
        ax1.set_ylim(0, 1.05)
        ax1.legend(loc="upper right", facecolor="#1a1d24", edgecolor="#444", labelcolor="#eee", fontsize=8)

        ax2 = self.figure.add_subplot(gs[1])
        ax2.set_facecolor("#11141a")
        ax2.set_title("CIE 1931 Chromaticity Diagram", color="#eee")
        ax2.set_xlabel("x", color="#eee")
        ax2.set_ylabel("y", color="#eee")
        ax2.tick_params(colors="#bbb")
        for s in ax2.spines.values(): s.set_color("#555")
        ax2.grid(True, color="#333", alpha=0.4)

        for i in range(len(LOCUS_POINTS)-1):
            wl = WAVELENGTHS_NM[i]
            rgb = SpectralPhysics.wavelength_to_rgb(wl)
            ax2.plot(LOCUS_POINTS[i:i+2, 0], LOCUS_POINTS[i:i+2, 1], color=rgb, lw=3, solid_capstyle='round')
        ax2.fill(LOCUS_POINTS[:, 0], LOCUS_POINTS[:, 1], color="#8ad8ff", alpha=0.05)
        
        pl_x = [x for t, x, y in PLANCKIAN_LOCUS]
        pl_y = [y for t, x, y in PLANCKIAN_LOCUS]
        ax2.plot(pl_x, pl_y, color='white', lw=2, linestyle='--', label="Planckian Locus")
        for t, x, y in PLANCKIAN_LOCUS:
            if t in [3000, 4000, 5000, 6500, 10000]:
                ax2.plot(x, y, 'o', color='gray', markersize=4)
                ax2.text(x+0.01, y+0.01, f"{t}K", color='white', fontsize=8)
        
        ax2.plot(D65_X, D65_Y, 'o', color="white", markersize=6, label="D65 White")

        xy_a = self._get_xy(spd_a)
        xy_b = self._get_xy(spd_b)
        xy_m = self._get_xy(mixed_norm)

        ax2.plot([xy_a[0], xy_b[0]], [xy_a[1], xy_b[1]], '--', color="#888", lw=1.5)
        ax2.plot(xy_a[0], xy_a[1], 'o', color=self.state.color_a.name(), markersize=10, label="Color A")
        ax2.plot(xy_b[0], xy_b[1], 'o', color=self.state.color_b.name(), markersize=10, label="Color B")
        ax2.plot(xy_m[0], xy_m[1], '*', color=mixed_rgb, markersize=14, label="Mixed", markeredgecolor="black")
        
        ax2.set_xlim(-0.1, 0.8)
        ax2.set_ylim(-0.1, 0.9)
        ax2.set_aspect('equal', adjustable='box')
        ax2.legend(loc="upper right", facecolor="#1a1d24", edgecolor="#444", labelcolor="#eee", fontsize=8)

        self.canvas.draw()

    @staticmethod
    def _get_xy(spd: np.ndarray) -> tuple:
        X, Y, Z = SpectralPhysics.spd_to_xyz(spd, normalize_y=False)
        total = X + Y + Z
        if total < 1e-12: return (0.33, 0.33)
        return (X / total, Y / total)


# ==============================================================================
# SECTION 6 — TAB 2: VIDEO GAME EXAMPLES
# ==============================================================================
SCENARIOS = [
    {
        "name": "Halo: Plasma Grenade",
        "desc": "<b>Physics:</b> Additive mixing of pure lasers.<br><br>Simulates the intense glow of a Covenant plasma grenade. Combines a narrowband blue plasma trail (445nm) with a green indicator light (532nm). The resulting cyan is so spectrally pure that it exceeds the sRGB display gamut, causing the engine to clamp the color.",
        "preset_a": "Blue Laser (445nm)",
        "preset_b": "Green Laser (532nm)",
        "additive": True,
        "thickness": 1.0
    },
    {
        "name": "Counter-Strike: Flashbang",
        "desc": "<b>Physics:</b> Overexposed Additive Daylight.<br><br>Simulates the blinding flash of a stun grenade. Adding two intense broad-spectrum D65 whites results in a massive spike in luminance (Y > 1.0). The physical intensity is blinding, though the monitor can only display it as pure white (L*=100).",
        "preset_a": "D65 Daylight",
        "preset_b": "D65 Daylight",
        "additive": True,
        "thickness": 1.0
    },
    {
        "name": "Resident Evil: Chemical Pool",
        "desc": "<b>Physics:</b> Subtractive Beer-Lambert filtering.<br><br>Simulates a flashlight (3000K Tungsten) shining through a vat of magenta-tinted chemical waste. The thick magenta filter (thickness 5.0) strips away the green/yellow light from the tungsten bulb, leaving a deep, dark crimson red.",
        "preset_a": "Blackbody 3000K (Tungsten)",
        "preset_b": "Magenta Notch Filter",
        "additive": False,
        "thickness": 5.0
    },
    {
        "name": "Minecraft: Lava Underwater",
        "desc": "<b>Physics:</b> Subtractive filtering of Blackbody radiation.<br><br>Simulates viewing a hot lava source (approximated as 3000K blackbody) through a layer of water (Cyan filter, thickness 4.0). The water heavily absorbs the red/infrared radiation, transmitting only a faint, eerie blue-green glow.",
        "preset_a": "Blackbody 3000K (Tungsten)",
        "preset_b": "Cyan Filter (490nm)",
        "additive": False,
        "thickness": 4.0
    },
    {
        "name": "Mario: Glowing ? Block",
        "desc": "<b>Physics:</b> Additive narrowband + broadband.<br><br>Simulates the iconic golden Question Block. Adds the narrowband yellow of a sodium lamp (589nm) to broad-spectrum daylight. The resulting yellow is highly saturated and dominates the visual perception despite the white light base.",
        "preset_a": "Sodium Lamp (589nm)",
        "preset_b": "D65 Daylight",
        "additive": True,
        "thickness": 1.0
    }
]

class VideoGameExamplesTab(QWidget):
    def __init__(self, lab_tab: ColorimetryLab):
        super().__init__()
        self.lab = lab_tab
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Left panel: Scenario list
        left_panel = QVBoxLayout()
        lbl_title = QLabel("Select a Video Game Lighting Scenario:")
        lbl_title.setStyleSheet("font-weight: bold; font-size: 12pt; color: #8ad8ff;")
        left_panel.addWidget(lbl_title)
        
        self.scenario_list = QListWidget()
        for s in SCENARIOS:
            self.scenario_list.addItem(s["name"])
        self.scenario_list.currentRowChanged.connect(self.load_scenario)
        left_panel.addWidget(self.scenario_list)
        
        # Right panel: Details
        right_panel = QVBoxLayout()
        
        self.desc_browser = QTextBrowser()
        self.desc_browser.setOpenExternalLinks(False)
        
        swatch_row = QHBoxLayout()
        self.sw_a = make_swatch("Channel A")
        self.sw_b = make_swatch("Channel B")
        self.sw_m = make_swatch("Mixed Result")
        for s in (self.sw_a, self.sw_b, self.sw_m): swatch_row.addWidget(s)
        
        self.lbl_metrics = QLabel("L*: 0.0  |  a*: 0.0  |  b*: 0.0\nCCT: 0 K\nGamut: Valid")
        self.lbl_metrics.setStyleSheet("font-family: monospace; color: #ccc; font-size: 12pt; padding: 10px;")
        self.lbl_metrics.setAlignment(Qt.AlignCenter)
        
        self.btn_load = QPushButton("Load Scenario into Colorimetry Lab →")
        self.btn_load.setStyleSheet("font-weight: bold; padding: 10px; font-size: 11pt;")
        self.btn_load.clicked.connect(self.send_to_lab)
        
        right_panel.addWidget(QLabel("Scenario Details:"))
        right_panel.addWidget(self.desc_browser)
        right_panel.addLayout(swatch_row)
        right_panel.addWidget(self.lbl_metrics)
        right_panel.addStretch(1)
        right_panel.addWidget(self.btn_load)
        
        layout.addLayout(left_panel, 1)
        layout.addLayout(right_panel, 2)
        
        self.scenario_list.setCurrentRow(0)

    def load_scenario(self, row):
        if row < 0: return
        sc = SCENARIOS[row]
        
        self.desc_browser.setText(sc["desc"])
        
        spd_a = SpectralPhysics.get_preset_spd(sc["preset_a"])
        spd_b = SpectralPhysics.get_preset_spd(sc["preset_b"])
        
        if sc["additive"]:
            mixed = SpectralPhysics.additive_mix(spd_a, spd_b)
        else:
            t_a = SpectralPhysics.beer_lambert_transmittance(spd_a, sc["thickness"])
            t_b = SpectralPhysics.beer_lambert_transmittance(spd_b, sc["thickness"])
            mixed = SpectralPhysics.subtractive_mix(t_a, t_b)
            
        m = mixed.max()
        mixed_norm = mixed / m if m > 1e-12 else mixed
        
        rgb_a, _ = SpectralPhysics.spd_to_rgb(spd_a)
        rgb_b, _ = SpectralPhysics.spd_to_rgb(spd_b)
        rgb_m, is_in_gamut = SpectralPhysics.spd_to_rgb(mixed_norm)
        
        X, Y, Z = SpectralPhysics.spd_to_xyz(mixed_norm, normalize_y=False)
        if Y > 1e-12:
            L, a, b = SpectralPhysics.xyz_to_lab(X/Y, 1.0, Z/Y)
        else:
            L, a, b = 0, 0, 0
        cct = SpectralPhysics.calculate_cct(X, Y, Z)
        
        set_swatch_color(self.sw_a, rgb_a, sc["preset_a"].split("(")[0])
        set_swatch_color(self.sw_b, rgb_b, sc["preset_b"].split("(")[0])
        h = "#{:02X}{:02X}{:02X}".format(int(rgb_m[0]*255), int(rgb_m[1]*255), int(rgb_m[2]*255))
        set_swatch_color(self.sw_m, rgb_m, h)
        
        gamut_str = "Valid (sRGB)" if is_in_gamut else "WARNING (Clamped)"
        cct_str = f"{int(cct)} K" if not np.isnan(cct) else "N/A"
        self.lbl_metrics.setText(f"L*: {L:.1f}  |  a*: {a:.1f}  |  b*: {b:.1f}\nCCT: {cct_str}\nGamut: {gamut_str}")

    def send_to_lab(self):
        row = self.scenario_list.currentRow()
        if row < 0: return
        sc = SCENARIOS[row]
        
        self.lab.state.preset_a = sc["preset_a"]
        self.lab.state.preset_b = sc["preset_b"]
        self.lab.state.additive = sc["additive"]
        self.lab.state.thickness = sc["thickness"]
        
        self.lab.combo_a.setCurrentText(sc["preset_a"])
        self.lab.combo_b.setCurrentText(sc["preset_b"])
        self.lab.mode_combo.setCurrentIndex(0 if sc["additive"] else 1)
        self.lab.thick_slider.setValue(int(sc["thickness"] * 10))
        
        self.window().tabs.setCurrentWidget(self.lab)


# ==============================================================================
# SECTION 7 — MAIN WINDOW
# ==============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectral Fourier Engine — Pro Colorimetry Suite")
        self.resize(1400, 900)
        self.setStyleSheet(DARK_QSS)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.lab_tab = ColorimetryLab()
        self.game_tab = VideoGameExamplesTab(self.lab_tab)

        self.tabs.addTab(self.lab_tab, "Colorimetry Lab")
        self.tabs.addTab(self.game_tab, "Video Game Examples")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor("#1a1d24"))
    dark_palette.setColor(QPalette.WindowText, QColor("#e0e0e0"))
    dark_palette.setColor(QPalette.Base, QColor("#11141a"))
    dark_palette.setColor(QPalette.AlternateBase, QColor("#1a1d24"))
    dark_palette.setColor(QPalette.Text, QColor("#e0e0e0"))
    dark_palette.setColor(QPalette.Button, QColor("#2a2e38"))
    app.setPalette(dark_palette)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())