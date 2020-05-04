#pragma once

using Real = double;

#define MpiReal MPI_DOUBLE_PRECISION
#define MpiComp MPI_C_DOUBLE_COMPLEX

constexpr Real Pi = 3.1415926535897932384626433832795;
constexpr Real _2Pi = 6.283185307179586476925286766559;

// DO NOT USE CLANG
#ifdef __clang__
#define CexSin(...)   (__VA_ARGS__, (Real) 0)
#define CexCos(...)   (__VA_ARGS__, (Real) 0)
#define CexExp(...)   (__VA_ARGS__, (Real) 0)
#define CexLdExp(...) (__VA_ARGS__, (Real) 0)
#else
#define CexSin    __builtin_sin
#define CexCos    __builtin_cos
#define CexExp    __builtin_exp
#define CexLdExp  __builtin_ldexp
#endif

struct Comp {
  Real re, im;

  Comp() noexcept = default;
  constexpr Comp(const Comp&) noexcept = default;
  constexpr Comp(Comp&&) noexcept = default;

  constexpr Comp(Real re, Real im = 0) : re(re), im(im) {}

  constexpr Comp& operator =(const Comp&) noexcept = default;
  constexpr Comp& operator =(Comp&&) noexcept = default;

  constexpr Comp operator +() const noexcept { return *this; }
  constexpr Comp operator -() const noexcept { return {-re, -im}; }

  constexpr Comp& operator +=(const Comp& z) noexcept {
    re += z.re;
    im += z.im;
    return *this;
  }

  constexpr Comp& operator -=(const Comp& z) noexcept {
    re -= z.re;
    im -= z.im;
    return *this;
  }

  constexpr Comp& operator *=(const Comp& t) noexcept {
    return *this = *this * t;
  }

  constexpr Comp& operator /=(const Comp& t) noexcept {
    return *this = *this / t;
  }

  constexpr Comp& operator +=(Real t) noexcept {
    re += t;
    return *this;
  }

  constexpr Comp& operator -=(Real t) noexcept {
    re -= t;
    return *this;
  }

  constexpr Comp& operator *=(Real t) noexcept {
    re *= t;
    im *= t;
    return *this;
  }

  constexpr Comp& operator /=(Real t) noexcept {
    re /= t;
    im /= t;
    return *this;
  }

  friend constexpr Comp operator +(const Comp& a, const Comp& b) noexcept {
    return {a.re + b.re, a.im + b.im};
  }
  friend constexpr Comp operator -(const Comp& a, const Comp& b) noexcept {
    return {a.re - b.re, a.im - b.im};
  }
  friend constexpr Comp operator *(const Comp& a, const Comp& b) noexcept {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
  }
  friend constexpr Comp operator /(const Comp& a, const Comp& b) noexcept {
    auto d = b.re * b.re + b.im * b.im;
    return {(a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d};
  }
  friend constexpr Comp operator +(const Comp& a, Real b) noexcept {
    return {a.re + b, a.im};
  }
  friend constexpr Comp operator -(const Comp& a, Real b) noexcept {
    return {a.re - b, a.im};
  }
  friend constexpr Comp operator *(const Comp& a, Real b) noexcept {
    return {a.re * b, a.im * b};
  }
  friend constexpr Comp operator /(const Comp& a, Real b) noexcept {
    return {a.re / b, a.im / b};
  }
  friend constexpr Comp operator +(Real a, const Comp& b) noexcept {
    return {a + b.re, b.im};
  }
  friend constexpr Comp operator -(Real a, const Comp& b) noexcept {
    return {a - b.re, -b.im};
  }
  friend constexpr Comp operator *(Real a, const Comp& b) noexcept {
    return {a * b.re, a * b.im};
  }
  friend constexpr Comp operator /(Real a, const Comp& b) noexcept {
    auto d = b.re * b.re + b.im * b.im;
    return {a * b.re / d, -a * b.im / d};
  }

  constexpr Comp ldexp(int e) noexcept {
    return {CexLdExp(re, e), CexLdExp(im, e)};
  }
};

constexpr Comp conj(const Comp& z) noexcept {
  return {z.re, -z.im};
}

constexpr Comp cis(Real t) noexcept {
  return {CexCos(t), CexSin(t)};
}

constexpr Comp exp(const Comp& z) noexcept {
  return cis(z.im) * CexExp(z.re);
}

template<class I>
constexpr I ceil2(I x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  if (sizeof(I) > 1)
    x |= x >> 8;
  if (sizeof(I) > 2)
    x |= x >> 16;
  if (sizeof(I) > 4)
    x |= x >> 32;
  return x + 1;
}
