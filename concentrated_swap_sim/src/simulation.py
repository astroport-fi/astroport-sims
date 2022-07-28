# source from: https://github.com/curvefi/curve-crypto-contract/blob/master/tests/simulation_int_many.py

A_MULTIPLIER = 10000


def geometric_mean(x):
    N = len(x)
    x = sorted(x, reverse=True)  # Presort - good for convergence
    D = x[0]
    for i in range(255):
        D_prev = D
        tmp = 10 ** 18
        for _x in x:
            tmp = tmp * _x // D
        D = D * ((N - 1) * 10 ** 18 + tmp) // (N * 10 ** 18)
        diff = abs(D - D_prev)
        if diff <= 1 or diff * 10 ** 18 < D:
            return D
    # print(x)
    raise ValueError("Did not converge")


def reduction_coefficient(x, gamma):
    N = len(x)
    # x_prod = 10 ** 18
    K = 10 ** 18
    S = sum(x)
    for x_i in x:
        # x_prod = x_prod * x_i // 10 ** 18
        K = K * N * x_i // S
    if gamma > 0:
        K = gamma * 10 ** 18 // (gamma + 10 ** 18 - K)
    return K


def newton_D(A, gamma, x, D0):
    D = D0
    i = 0

    S = sum(x)
    x = sorted(x, reverse=True)
    N = len(x)

    for i in range(255):
        D_prev = D

        K0 = 10 ** 18
        for _x in x:
            K0 = K0 * _x * N // D

        _g1k0 = abs(gamma + 10 ** 18 - K0)

        # D / (A * N**N) * _g1k0**2 / gamma**2
        mul1 = 10 ** 18 * D // gamma * _g1k0 // gamma * _g1k0 * A_MULTIPLIER // A

        # 2*N*K0 / _g1k0
        mul2 = (2 * 10 ** 18) * N * K0 // _g1k0

        neg_fprime = (S + S * mul2 // 10 ** 18) + mul1 * N // K0 - mul2 * D // 10 ** 18
        assert neg_fprime > 0  # Python only: -f' > 0

        # D -= f / fprime
        D = (D * neg_fprime + D * S - D ** 2) // neg_fprime - D * (mul1 // neg_fprime) // 10 ** 18 * (
                10 ** 18 - K0) // K0

        if D < 0:
            D = -D // 2
        if abs(D - D_prev) <= max(100, D // 10 ** 14):
            return D

    raise ValueError("Did not converge")


N_COINS = 2


def unsafe_div(a, b):
    return a // b


def unsafe_sub(a, b):
    return a - b


def newton_y_vyper(ANN, gamma, x, D, i):
    """Ported from Vyper code"""

    x_j = x[1 - i]
    y = D ** 2 // (x_j * N_COINS ** 2)
    K0_i = (10 ** 18 * N_COINS) * x_j // D
    # S_i = x_j

    # frac = x_j * 1e18 / D => frac = K0_i / N_COINS
    assert (K0_i > 10 ** 16 * N_COINS - 1) and (K0_i < 10 ** 20 * N_COINS + 1)  # dev: unsafe values x[i]

    # x_sorted[N_COINS] = x
    # x_sorted[i] = 0
    # x_sorted = self.sort(x_sorted)  # From high to low
    # x[not i] instead of x_sorted since x_soted has only 1 element

    convergence_limit = max(max(x_j // 10 ** 14, D // 10 ** 14), 100)

    __g1k0 = gamma + 10 ** 18

    for j in range(255):
        y_prev = y

        K0 = unsafe_div(K0_i * y * N_COINS, D)
        S = x_j + y

        _g1k0 = __g1k0
        if _g1k0 > K0:
            _g1k0 = unsafe_sub(_g1k0, K0) + 1
        else:
            _g1k0 = unsafe_sub(K0, _g1k0) + 1

        # D / (A * N**N) * _g1k0**2 / gamma**2
        mul1 = unsafe_div(unsafe_div(unsafe_div(10 ** 18 * D, gamma) * _g1k0, gamma) * _g1k0 * A_MULTIPLIER, ANN)

        # 2*K0 / _g1k0
        mul2 = unsafe_div(10 ** 18 + (2 * 10 ** 18) * K0, _g1k0)

        yfprime = 10 ** 18 * y + S * mul2 + mul1
        _dyfprime = D * mul2
        if yfprime < _dyfprime:
            y = unsafe_div(y_prev, 2)
            continue
        else:
            yfprime = unsafe_sub(yfprime, _dyfprime)
        fprime = yfprime // y

        # y -= f / f_prime;  y = (y * fprime - f) / fprime
        # y = (yfprime + 10**18 * D - 10**18 * S) // fprime + mul1 // fprime * (10**18 - K0) // K0
        y_minus = mul1 // fprime
        y_plus = (yfprime + 10 ** 18 * D) // fprime + y_minus * 10 ** 18 // K0
        y_minus += 10 ** 18 * S // fprime

        if y_plus < y_minus:
            y = unsafe_div(y_prev, 2)
        else:
            y = unsafe_sub(y_plus, y_minus)

        if y > y_prev:
            diff = unsafe_sub(y, y_prev)
        else:
            diff = unsafe_sub(y_prev, y)
        if diff < max(convergence_limit, unsafe_div(y, 10 ** 14)):
            frac = unsafe_div(y * 10 ** 18, D)
            assert (frac > 10 ** 16 - 1) and (frac < 10 ** 20 + 1)  # dev: unsafe value for y
            return y

    raise Exception("Did not converge")


def newton_y(A, gamma, x, D, i):
    N = len(x)

    y = D // N
    K0_i = 10 ** 18
    S_i = 0
    x_sorted = sorted(_x for j, _x in enumerate(x) if j != i)
    convergence_limit = max(max(x_sorted) // 10 ** 14, D // 10 ** 14, 100)
    for _x in x_sorted:
        y = y * D // (_x * N)  # Small _x first
        S_i += _x
    for _x in x_sorted[::-1]:
        K0_i = K0_i * _x * N // D  # Large _x first

    for j in range(255):
        y_prev = y

        K0 = K0_i * y * N // D
        S = S_i + y

        _g1k0 = abs(gamma + 10 ** 18 - K0)

        # D / (A * N**N) * _g1k0**2 / gamma**2
        mul1 = 10 ** 18 * D // gamma * _g1k0 // gamma * _g1k0 * A_MULTIPLIER // A

        # 2*K0 / _g1k0
        mul2 = 10 ** 18 + (2 * 10 ** 18) * K0 // _g1k0

        yfprime = (10 ** 18 * y + S * mul2 + mul1 - D * mul2)
        fprime = yfprime // y
        assert fprime > 0  # Python only: f' > 0

        # y -= f / f_prime;  y = (y * fprime - f) / fprime
        y = (yfprime + 10 ** 18 * D - 10 ** 18 * S) // fprime + mul1 // fprime * (10 ** 18 - K0) // K0

        # if j > 100:  # Just logging when doesn't converge
        #     print(j, y, D, x)
        if y < 0 or fprime < 0:
            y = y_prev // 2
        if abs(y - y_prev) <= max(convergence_limit, y // 10 ** 14):
            return y

    raise Exception("Did not converge")


def solve_x(A, gamma, x, D, i):
    # return newton_y(A, gamma, x, D, i)
    return newton_y_vyper(A, gamma, x, D, i)


def solve_D(A, gamma, x):
    D0 = len(x) * geometric_mean(x)  # <- fuzz to make sure it's ok XXX
    return newton_D(A, gamma, x, D0)


def solve_D_vyper(ANN, gamma, x_unsorted):
    """
    Finding the invariant using Newton method.
    ANN is higher by the factor A_MULTIPLIER
    ANN is already A * N**N

    Currently uses 60k gas
    """

    # Initial value of invariant D is that for constant-product invariant
    x = x_unsorted
    if x[0] < x[1]:
        x = [x_unsorted[1], x_unsorted[0]]

    assert x[0] > 10 ** 9 - 1 and x[0] < 10 ** 15 * 10 ** 18 + 1  # dev: unsafe values x[0]
    assert x[1] * 10 ** 18 / x[0] > 10 ** 14 - 1  # dev: unsafe values x[i] (input)

    D = N_COINS * geometric_mean(x)
    S = x[0] + x[1]
    __g1k0 = gamma + 10 ** 18

    for i in range(255):
        D_prev = D
        assert D > 0
        # Unsafe division by D is now safe

        # K0 = 10**18
        # for _x in x:
        #     K0 = K0 * _x * N_COINS / D
        # collapsed for 2 coins
        K0 = unsafe_div(unsafe_div((10 ** 18 * N_COINS ** 2) * x[0], D) * x[1], D)

        _g1k0 = __g1k0
        if _g1k0 > K0:
            _g1k0 = unsafe_sub(_g1k0, K0) + 1  # > 0
        else:
            _g1k0 = unsafe_sub(K0, _g1k0) + 1  # > 0

        # D / (A * N**N) * _g1k0**2 / gamma**2
        mul1 = unsafe_div(unsafe_div(unsafe_div(10 ** 18 * D, gamma) * _g1k0, gamma) * _g1k0 * A_MULTIPLIER, ANN)

        # 2*N*K0 / _g1k0
        mul2 = unsafe_div(((2 * 10 ** 18) * N_COINS) * K0, _g1k0)

        neg_fprime = (S + unsafe_div(S * mul2, 10 ** 18)) + mul1 * N_COINS // K0 - unsafe_div(mul2 * D, 10 ** 18)

        # D -= f / fprime
        D_plus = D * (neg_fprime + S) // neg_fprime
        D_minus = D * D // neg_fprime
        if 10 ** 18 > K0:
            D_minus += unsafe_div(D * (mul1 // neg_fprime), 10 ** 18) * unsafe_sub(10 ** 18, K0) // K0
        else:
            D_minus -= unsafe_div(D * (mul1 // neg_fprime), 10 ** 18) * unsafe_sub(K0, 10 ** 18) // K0

        if D_plus > D_minus:
            D = unsafe_sub(D_plus, D_minus)
        else:
            D = unsafe_div(unsafe_sub(D_minus, D_plus), 2)

        if D > D_prev:
            diff = unsafe_sub(D, D_prev)
        else:
            diff = unsafe_sub(D_prev, D)

        if diff * 10 ** 14 < max(10 ** 16, D):  # Could reduce precision for gas efficiency here
            # Test that we are safe with the next newton_y
            for _x in x:
                frac = _x * 10 ** 18 // D
                assert (frac > 10 ** 16 - 1) and (frac < 10 ** 20 + 1)  # dev: unsafe values x[i]
            return D

    raise Exception("Did not converge")


class Curve:
    def __init__(self, A, gamma, D, n, p=None):
        self.A = A
        self.gamma = gamma
        self.n = n
        if p:
            self.p = p
        else:
            self.p = [10 ** 18] * n
        # These lines differ from the original Curve's code.
        # We need to have an ability to define initial pool balances
        if isinstance(D, list):
            self.x = D
        else:
            self.x = [D // n * 10 ** 18 // _p for _p in self.p]

    def xp(self):
        return [x * p // 10 ** 18 for x, p in zip(self.x, self.p)]

    def D(self, xp=None):
        xp = xp or self.xp()
        if any(x <= 0 for x in xp):
            raise ValueError
        # return solve_D(self.A, self.gamma, xp)
        return solve_D_vyper(self.A, self.gamma, xp)

    def y(self, x, i, j, hack=(None, None)):
        xp, dy = hack
        xp = xp or self.xp()
        xp[i] = x * self.p[i] // 10 ** 18
        old_xp = None
        if dy:
            old_xp = xp.copy()
            old_xp[i] = old_xp[i] - dy
        yp = solve_x(self.A, self.gamma, xp, self.D(xp=old_xp), j)
        return yp


EXP_PRECISION = 1


def half_pow(power: int) -> int:
    """port from Vyper code"""

    intpow = power // 10 ** 18
    if intpow > 59:
        return 0
    otherpow = power - intpow * 10 ** 18  # < 10**18
    result = 2 ** intpow
    result = 10 ** 18 // result
    if otherpow == 0:
        return result

    term = 10 ** 18
    S = 10 ** 18
    neg: bool = False

    for i in range(1, 64):
        K = i * 10 ** 18  # <= 255 * 10**18; >= 10**18
        c = K - 10 ** 18  # <= 254 * 10**18; < K
        if otherpow > c:  # c < otherpow < 10**18 <= K -> c < K
            c = otherpow - c
            neg = not neg
        else:
            c = c - otherpow  # c < K
        # c <= 254 * 10**18, < K -> (c/2) / K < 1 -> term * c/2 / K <= 10**18
        term = (term * (c // 2)) // K
        if neg:
            S -= term
        else:
            S += term
        if term < EXP_PRECISION:
            return result * S // 10 ** 18

    raise "Did not converge"


class Trader:
    def __init__(self, A, gamma, D, n, p0, mid_fee=1e-3, out_fee=3e-3, allowed_extra_profit=2 * 10 ** 13,
                 fee_gamma=None,
                 adjustment_step=0.003, ma_half_time=500, log=True):
        # allowed_extra_profit is actually not used
        self.p0 = p0[:]
        self.price_oracle = self.p0[:]
        self.last_price = self.p0[:]
        self.curve = Curve(A, gamma, D, n, p=p0[:])
        self.dx = int(sum(D) * 1e-8)
        self.mid_fee = int(mid_fee * 1e10)
        self.out_fee = int(out_fee * 1e10)
        self.D0 = self.curve.D()
        self.xcp_0 = self.get_xcp()
        self.xcp_profit = 10 ** 18
        self.xcp_profit_real = 10 ** 18
        self.xcp = self.xcp_0
        self.allowed_extra_profit = allowed_extra_profit
        self.adjustment_step = int(10 ** 18 * adjustment_step)
        self.log = log
        self.fee_gamma = fee_gamma or gamma
        self.total_vol = 0.0
        self.ma_half_time = ma_half_time
        self.ext_fee = 0  # 0.03e-2
        self.slippage = 0
        self.slippage_count = 0

        self.t = 0

    def fee(self, xp=None):
        f = reduction_coefficient(xp or self.curve.xp(), self.fee_gamma)
        return (self.mid_fee * f + self.out_fee * (10 ** 18 - f)) // 10 ** 18

    def price(self, i, j):
        dx_raw = self.dx * 10 ** 18 // self.curve.p[i]
        return dx_raw * 10 ** 18 // (self.curve.x[j] - self.curve.y(self.curve.x[i] + dx_raw, i, j))

    def step_for_price(self, dp, pair, sign=1):
        a, b = pair
        p0 = self.price(*pair)
        dp = p0 * dp // 10 ** 18
        x0 = self.curve.x[:]
        step = self.dx * 10 ** 18 // self.curve.p[a]
        while True:
            self.curve.x[a] = x0[a] + sign * step
            dp_ = abs(p0 - self.price(*pair))
            if dp_ >= dp or step >= self.curve.x[a] // 10:
                self.curve.x = x0
                return step
            step *= 2

    def get_xcp(self):
        # First calculate the ideal balance
        # Then calculate, what the constant-product would be
        D = self.curve.D()
        N = len(self.curve.x)
        X = [D * 10 ** 18 // (N * p) for p in self.curve.p]

        return geometric_mean(X)

    def update_xcp(self, only_real=False, xcp=None):
        xcp = xcp or self.get_xcp()
        self.xcp_profit_real = self.xcp_profit_real * xcp // self.xcp
        if not only_real:
            self.xcp_profit = self.xcp_profit * xcp // self.xcp
        self.xcp = xcp

    def buy(self, dx, i, j, max_price=1e100):
        """
        Buy y for x
        """
        try:
            x_old = self.curve.x[:]
            x = self.curve.x[i] + dx
            y = self.curve.y(x, i, j)
            dy = self.curve.x[j] - y
            self.curve.x[i] = x
            self.curve.x[j] = y
            fee = self.fee()
            self.curve.x[j] += dy * fee // 10 ** 10
            dy = dy * (10 ** 10 - fee) // 10 ** 10
            if dx * 10 ** 18 // dy > max_price or dy < 0:
                self.curve.x = x_old
                return False
            self.update_xcp()
            return dy
        except ValueError:
            return False

    def sell(self, dy, i, j, min_price=0):
        """
        Sell y for x
        """
        try:
            x_old = self.curve.x[:]

            y = self.curve.x[j] + dy

            # hacking internal pools' representation
            x = self.curve.x.copy()
            x[j] += dy
            hacked_xp = [x * p // 10 ** 18 for x, p in zip(x, self.curve.p)]
            hacked_xp[j] -= dy * self.curve.p[j] // 10 ** 18

            # calculate swap using internal pools' representation
            internal_x = self.curve.y(y, j, i, hack=(hacked_xp, dy * self.curve.p[j] // 10 ** 18))
            internal_dx = hacked_xp[i] - internal_x
            hacked_xp[i] = internal_x - internal_dx
            fee = self.fee(xp=hacked_xp)
            fee_amount = internal_dx * fee // 10 ** 10
            hacked_xp[i] += fee_amount
            internal_dx -= fee_amount

            dx = internal_dx * 10 ** 18 // self.curve.p[i]
            self.curve.x[i] = x_old[i] - dx
            self.curve.x[j] = y
            if dx * 10 ** 18 // dy < min_price or dx < 0:
                self.curve.x = x_old
                return False
            # self.update_xcp()
            return dx
        except ValueError:
            return False

    def ma_recorder(self, t, price_vector):
        # XXX what if every block only has p_b being last
        N = len(price_vector)
        if t > self.t:
            arg = (t - self.t) * int(1e18) // self.ma_half_time
            alpha = half_pow(arg)
            for k in range(1, N):
                self.price_oracle[k] = \
                    (price_vector[k] * (int(1e18) - alpha) + self.price_oracle[k] * alpha) // int(1e18)
            self.t = t

    def tweak_price(self, t, a, b, p, debug=False):
        self.ma_recorder(t, self.last_price)
        if b > 0:
            self.last_price[b] = p * self.last_price[a] // 10 ** 18
        else:
            self.last_price[a] = self.last_price[0] * 10 ** 18 // p

        # simplified for 2pool
        norm = abs(self.price_oracle[1] * 10 ** 18 // self.curve.p[1] - 10 ** 18)
        adjustment_step = max(self.adjustment_step, norm // 10)
        if norm <= adjustment_step:
            # Already close to the target price
            return norm

        p_new = [10 ** 18]
        p_new += [p_target + adjustment_step * (p_real - p_target) // norm
                  for p_real, p_target in zip(self.price_oracle[1:], self.curve.p[1:])]

        old_p = self.curve.p[:]
        old_profit_real = self.xcp_profit_real
        old_profit = self.xcp_profit
        old_xcp = self.xcp

        self.update_xcp()
        _xp = self.curve.xp()
        _xp[1] = _xp[1] * p_new[1] // old_p[1]
        _D = solve_D_vyper(self.curve.A, self.curve.gamma, _xp)
        _xp = [_D // N_COINS, _D * 10 ** 18 // (N_COINS * p_new[1])]
        self.update_xcp(only_real=True, xcp=geometric_mean(_xp))
        self.curve.p = p_new

        if 2 * (self.xcp_profit_real - 10 ** 18) <= self.xcp_profit - 10 ** 18:
            # If real profit is less than half of maximum - revert params back
            self.curve.p = old_p
            self.xcp_profit_real = old_profit_real
            self.xcp_profit = old_profit
            self.xcp = old_xcp
        else:
            if debug:
                print(f"Adjusted price: {old_p[1]} -> {self.curve.p[1]}")

        return norm

    def simulate(self, mdata):
        lasts = {}
        self.t = mdata[0]['t']
        for i, d in enumerate(mdata):
            a, b = d['pair']
            vol = 0
            ext_vol = int(d['volume'] * self.price_oracle[b])  # <- now all is in USD
            ctr = 0
            last = lasts.get((a, b), self.price_oracle[b] * 10 ** 18 // self.price_oracle[a])
            _high = last
            _low = last

            # Dynamic step
            # f = reduction_coefficient(self.curve.xp(), self.curve.gamma)
            candle = min(int(1e18 * abs((d['high'] - d['low']) / d['high'])), 10 ** 17)
            candle = max(10 ** 15, candle)
            step1 = self.step_for_price(candle // 50, (a, b), sign=1)
            step2 = self.step_for_price(candle // 50, (a, b), sign=-1)
            step = min(step1, step2)

            max_price = int(1e18 * d['high'])
            _dx = 0
            p_before = self.price(a, b)
            while last < max_price and vol < ext_vol // 2:
                dy = self.buy(step, a, b, max_price=max_price)
                if dy is False:
                    break
                vol += dy * self.price_oracle[b] // 10 ** 18
                _dx += dy
                last = step * 10 ** 18 // dy
                max_price = int(1e18 * d['high'])
                ctr += 1
            p_after = self.price(a, b)
            if p_before != p_after:
                self.slippage_count += 1
                self.slippage += _dx * self.curve.p[b] // 10 ** 18 * (p_before + p_after) // (
                        2 * abs(p_before - p_after))
            _high = last
            min_price = int(1e18 * d['low'])
            _dx = 0
            p_before = p_after
            while last > min_price and vol < ext_vol // 2:
                dx = step * 10 ** 18 // last
                dy = self.sell(dx, a, b, min_price=min_price)
                _dx += dx
                if dy is False:
                    break
                vol += dx * self.price_oracle[b] // 10 ** 18
                last = dy * 10 ** 18 // dx
                min_price = int(10 ** 18 * d['low'])
                ctr += 1
            p_after = self.price(a, b)
            if p_before != p_after:
                self.slippage_count += 1
                self.slippage += _dx * self.curve.p[b] // 10 ** 18 * (p_before + p_after) // (
                        2 * abs(p_before - p_after))
            _low = last
            lasts[(a, b)] = last

            self.tweak_price(d['t'], a, b, (_high + _low) // 2)

            self.total_vol += vol
            if self.log:
                try:
                    print(("""{0:.1f}%\ttrades: {1}\t"""
                           """AMM: {2:.0f}, {3:.0f}\tTarget: {4:.0f}, {5:.0f}\t"""
                           """Vol: {6:.4f}\tPR:{7:.2f}\txCP-growth: {8:.5f}\t"""
                           """APY:{9:.1f}%\tfee:{10:.3f}%""").format(
                        100 * i / len(mdata), ctr,
                        lasts.get((0, 1), self.price_oracle[1] * 10 ** 18 // self.price_oracle[0]) / 1e18,
                        lasts.get((0, 2), self.price_oracle[2] * 10 ** 18 // self.price_oracle[0]) / 1e18,
                        self.curve.p[1] / 1e18,
                        self.curve.p[2] / 1e18,
                        self.total_vol / 1e18,
                        (self.xcp_profit_real - 10 ** 18) / (self.xcp_profit - 10 ** 18),
                        self.xcp_profit_real / 1e18,
                        ((self.xcp_profit_real / 1e18) ** (86400 * 365 / (d['t'] - mdata[0]['t'] + 1)) - 1) * 100,
                        self.fee() / 1e10 * 100))
                except Exception:
                    pass
