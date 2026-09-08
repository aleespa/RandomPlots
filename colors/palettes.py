GrnGre = ["#23895d", "#7fd677", "#eadb87", "#eeda9e", "#b8b8b8"]
GYRYG = ["#96ceb4", "#ffeead", "#ff6f69", "#ffcc5c", "#88d8b0"]
YBBPG = ["#f7f4a3", "#7fccec", "#6a81d9", "#a479c9", "#dfdfdf"]
RedWht = [
    "#ff817a",
    "#ff8d87",
    "#ff9a94",
    "#ffa6a1",
    "#ffb3af",
    "#ffc0bc",
    "#ffccc9",
    "#ffd9d7",
    "#ffe5e4",
    "#fff2f1",
    "#ffffff",
]
TWLAGN = ["#ccabd8", "#8474a1", "#6ec6ca", "#08979d", "#055b5c"]
NEAURA = ["#ff00b4", "#00ffbc", "#8ea5ff", "#ffffff", "#c493ff"]
ECOSPL = ["#e26000", "#228B46", "#5092B8", "#ff9b9b", "#c9d06c", "#22ba5a", "#58c0e7"]
SUNWAV = [
    ("#ff6f4b", 20),
    ("#fd4c55", 20),
    ("#e13661", 20),
    ("#c1246b", 20),
    ("#a11477", 20),
    ("#c1246b", 20),
    ("#e13661", 20),
    ("#fd4c55", 20),
    ("#ff6f4b", 40),
]
CYBERPUNK = ["#ff007f", "#00f0ff", "#ffe600", "#7b2cbf", "#ff5400", "#00ff87"]
ELECTRIC_AURORA = ["#00ff88", "#00e5ff", "#ff00aa", "#8338ec", "#ffbe0b", "#3a86ff"]
SOLAR_FLARE = ["#7209b7", "#f72585", "#ff3600", "#ff9e00", "#ffee32", "#ffffff"]
HOLOGRAPHIC = ["#06d6a0", "#118ab2", "#8338ec", "#ff006e", "#fb5607", "#ffbe0b"]

# --- Density ramps ------------------------------------------------------------
# Ordered, not categorical: feed these to LinearSegmentedColormap.from_list and
# index them with a normalised density. All share the same ten-stop rhythm --
# black, deep tint, dark hue, mid hue, pale hue, white, accent, hot accent, deep
# dark, black -- so the sparse haze takes the mid hue, the bright edge goes
# white, and the densest cores fall back through the accent into near-black.
CONJUGATE_BLOOM = [
    "#000000",
    "#2a0810",
    "#173a45",
    "#2fa0a8",
    "#a9ece9",
    "#fff6ee",
    "#ff6ae4",
    "#f600f3",
    "#3a0c08",
    "#000000",
]
# Orange haze, cyan spine -- the widest two-tone separation of the set.
EMBER_FURNACE = [
    "#000000",
    "#0b0716",
    "#3d1206",
    "#b03a08",
    "#ffb347",
    "#fff4e2",
    "#5df0ff",
    "#00b9ff",
    "#050d2a",
    "#000000",
]
# Jade haze, coral cores. The dimmest ramp here; lift the mid stop for punch.
VERDANT_DECAY = [
    "#000000",
    "#07130c",
    "#123d2a",
    "#2f9e6b",
    "#b8f0c2",
    "#fbfff2",
    "#ff8a5c",
    "#ff2e63",
    "#240610",
    "#000000",
]
# Near-monochrome sepia with a steel-blue accent that only surfaces in the cores.
NOIR_GOLD = [
    "#000000",
    "#120c05",
    "#3c2a10",
    "#9c6b1f",
    "#e6c169",
    "#fff8e7",
    "#9fd0e8",
    "#3f7fa8",
    "#0a1116",
    "#000000",
]
# Violet haze, mint spine -- NEAURA territory, a quarter-turn off CONJUGATE_BLOOM.
ULTRAVIOLET_BLOOM = [
    "#000000",
    "#0a0418",
    "#2a1057",
    "#6a3df0",
    "#c0b6ff",
    "#ffffff",
    "#7cffcf",
    "#00e5a0",
    "#04211a",
    "#000000",
]
# Built from SUNWAV's hexes: plum-to-coral haze over a teal spine.
SUNWAV_TIDE = [
    "#000000",
    "#1a0620",
    "#5c0a4a",
    "#c1246b",
    "#ff8f6b",
    "#fff1e0",
    "#9ef0ea",
    "#0f9ea6",
    "#08131e",
    "#000000",
]
# Ice-blue haze, warm gold reserved for the densest knots.
GLACIER = [
    "#000000",
    "#05101c",
    "#123a5c",
    "#3f8fc4",
    "#bfe4f7",
    "#ffffff",
    "#ffe8a8",
    "#e0a63c",
    "#160e04",
    "#000000",
]
# No hue until the cores: greyscale silk with a warm taupe centre.
BONE = [
    "#000000",
    "#0d0d0f",
    "#33343a",
    "#7d7f88",
    "#c9cbd2",
    "#ffffff",
    "#d9c9b4",
    "#6b5744",
    "#0a0806",
    "#000000",
]
