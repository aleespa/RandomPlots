import base64
import io
import math

import matplotlib.pyplot as plt
import numpy as np


def well(x):
    return 1 - 2 / (1 + x ** 2) ** 8


def tent(x):
    return 1 - 2 * np.abs(x)


class VariableX:
    arity = 0

    def __init__(self, rng):
        pass

    def __repr__(self):
        return "x"

    def eval(self, X, Y):
        return X, X, X


class VariableY:
    arity = 0

    def __init__(self, rng):
        pass

    def __repr__(self):
        return "y"

    def eval(self, X, Y):
        return Y, Y, Y


class Constant:
    arity = 0

    def __init__(self, rng):
        self.c = (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))

    def __repr__(self):
        return f'Constant{self.c}'

    def eval(self, X, Y):
        return (
            np.full_like(X, self.c[0], dtype=np.float32),
            np.full_like(X, self.c[1], dtype=np.float32),
            np.full_like(X, self.c[2], dtype=np.float32),
        )


class Sum:
    arity = 2

    def __init__(self, e1, e2, rng):
        self.e1, self.e2 = e1, e2

    def __repr__(self):
        return f'Sum({self.e1}, {self.e2})'

    def eval(self, X, Y):
        r1, g1, b1 = self.e1.eval(X, Y)
        r2, g2, b2 = self.e2.eval(X, Y)
        return (r1 + r2) / 2, (g1 + g2) / 2, (b1 + b2) / 2


class Product:
    arity = 2

    def __init__(self, e1, e2, rng):
        self.e1, self.e2 = e1, e2

    def __repr__(self):
        return f'Product({self.e1}, {self.e2})'

    def eval(self, X, Y):
        r1, g1, b1 = self.e1.eval(X, Y)
        r2, g2, b2 = self.e2.eval(X, Y)
        return r1 * r2, g1 * g2, b1 * b2


class Mod:
    arity = 2

    def __init__(self, e1, e2, rng):
        self.e1, self.e2 = e1, e2

    def __repr__(self):
        return f'Mod({self.e1}, {self.e2})'

    def eval(self, X, Y):
        r1, g1, b1 = self.e1.eval(X, Y)
        r2, g2, b2 = self.e2.eval(X, Y)
        with np.errstate(divide='ignore', invalid='ignore'):
            return (
                np.where(r2 != 0, np.mod(r1, r2), 0),
                np.where(g2 != 0, np.mod(g1, g2), 0),
                np.where(b2 != 0, np.mod(b1, b2), 0),
            )


class Well:
    arity = 1

    def __init__(self, e, rng):
        self.e = e

    def __repr__(self):
        return f'Well({self.e})'

    def eval(self, X, Y):
        r, g, b = self.e.eval(X, Y)
        return well(r), well(g), well(b)


class Tent:
    arity = 1

    def __init__(self, e, rng):
        self.e = e

    def __repr__(self):
        return f'Tent({self.e})'

    def eval(self, X, Y):
        r, g, b = self.e.eval(X, Y)
        return tent(r), tent(g), tent(b)


class Sin:
    arity = 1

    def __init__(self, e, rng):
        self.e = e
        self.phase = rng.uniform(0, math.pi)
        self.freq = rng.uniform(1.0, 12.0)

    def __repr__(self):
        return f'Sin({self.phase:.2f}+{self.freq:.2f}*{self.e})'

    def eval(self, X, Y):
        r, g, b = self.e.eval(X, Y)
        return (
            np.sin(self.phase + self.freq * r),
            np.sin(self.phase + self.freq * g),
            np.sin(self.phase + self.freq * b),
        )


class Level:
    arity = 3

    def __init__(self, level, e1, e2, rng):
        self.threshold = rng.uniform(-1, 1)
        self.level, self.e1, self.e2 = level, e1, e2

    def __repr__(self):
        return f'Level({self.threshold:.2f}, {self.level}, {self.e1}, {self.e2})'

    def eval(self, X, Y):
        lr, lg, lb = self.level.eval(X, Y)
        r1, g1, b1 = self.e1.eval(X, Y)
        r2, g2, b2 = self.e2.eval(X, Y)
        return (
            np.where(lr < self.threshold, r1, r2),
            np.where(lg < self.threshold, g1, g2),
            np.where(lb < self.threshold, b1, b2),
        )


class Mix:
    arity = 3

    def __init__(self, w, e1, e2, rng):
        self.w, self.e1, self.e2 = w, e1, e2

    def __repr__(self):
        return f'Mix({self.w}, {self.e1}, {self.e2})'

    def eval(self, X, Y):
        w, _, _ = self.w.eval(X, Y)
        weight = (w + 1) / 2
        r1, g1, b1 = self.e1.eval(X, Y)
        r2, g2, b2 = self.e2.eval(X, Y)
        return (
            weight * r1 + (1 - weight) * r2,
            weight * g1 + (1 - weight) * g2,
            weight * b1 + (1 - weight) * b2,
        )


operators = [
    VariableX,
    VariableY,
    Constant,
    Sum,
    Product,
    # Mod,
    # Well,
    # Tent,
    # Sin,
    Level,
    Mix,
]
operators0 = [op for op in operators if op.arity == 0]
operators1 = [op for op in operators if op.arity > 0]


def generate(k, rng):
    if k <= 0:
        return rng.choice(operators0)(rng)
    else:
        op = rng.choice(operators1)
        if op.arity > 1:
            splits = sorted(rng.choice(range(k), size=op.arity - 1, replace=True))
        else:
            splits = []
        args = []
        prev = 0
        for split in splits:
            args.append(generate(split - prev, rng))
            prev = split
        args.append(generate(k - 1 - prev, rng))
        return op(*args, rng=rng)


def generate_plot(seed, bg_color=(0, 0, 0), dark_mode=True):
    rng = np.random.default_rng(seed)
    size = 1024
    fig = plt.figure(figsize=(12, 12), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))

    x, y = np.meshgrid(
        np.linspace(-1, 1, size, dtype=np.float32),
        np.linspace(-1, 1, size, dtype=np.float32)
    )

    k = rng.integers(8, 15)
    expr = generate(k, rng)
    r, g, b = expr.eval(x, y)

    img = np.empty((size, size, 3), dtype=np.float32)
    np.add(r, 1, out=img[..., 0])
    np.add(g, 1, out=img[..., 1])
    np.add(b, 1, out=img[..., 2])
    np.clip(img, 0, 2, out=img)
    np.divide(img, 2, out=img)

    ax.imshow(img)
    ax.axis("off")
    buffer = io.BytesIO()
    plt.savefig(buffer, format='jpg', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)

    return buffer


def create_image(seed=0, dark_mode=True, bg_color=(0, 0, 0)):
    buffer = generate_plot(seed, bg_color, dark_mode)
    image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    return image_data
