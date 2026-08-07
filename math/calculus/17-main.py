#!/usr/bin/env python3

poly_integral = __import__('17-integrate').poly_integral

poly = [5, 3, 0, 1]
print(poly)
print(poly_integral(poly))

poly = [0]
print(poly)
print(poly_integral(poly))

poly = [0, 0, 0]
print(poly)
print(poly_integral(poly, 4))

poly = [0, 2, 0]
print(poly)
print(poly_integral(poly, 1))

poly = [0, 0]
print(poly)
print(poly_integral(poly, 7))
