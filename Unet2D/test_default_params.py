def tversky(a, b, c=0.2, d=0.3):
    print("a", a, "b", b, "c", c, "d", d)


def dice(a, b, c=0.2, d=0.3):
    print("[DICE] a", a, "b", b, "c", c, "d", d)


if __name__ == '__main__':
    x = 0.3
    y = 0.5

    alpha = 0.02
    beta = 0.98

    if x > 0.5:
        f = lambda a, b, c=alpha, d=beta: tversky(a, b, c, d)
    else:
        f = dice

    f(x, y)
