print(range(8,9))
print(list(range(0,10,1)))
print(list(range(1, 20, 1)))
print(list(range(10, 0, -1)))
# for i in range(7):
print(5/10)


import math
# 1. 
n = 10
print(math.log10(n)+1)

# 2.
print("",*range(1, n))
print("",*range(n, 0, -1))
print("",*range(2, n, 2))
print("",*range(1,n, 2))
print("",*range(1,n, 2))
# print("",*range(1,n, 2))
a = "print('hello world',sep='o')"
for i in a:
    print(i)
for i in a:
    print(ord(i))

def fd(n):
    fd = 0
    while n > 0:
        fd = n%10
        n//=10
    return fd

print("For First digit")

for i in range(49, 51):
    print(fd(i), " ")

# 1. 
n = 22
n = (1 + (n - 1)) % 9
print(n)

#2. 
print("reverse")
def reverse(n):
    sol = (n % 10)
    if sol > 0:
        print(sol,end=" ")
        return reverse(n//10)
    return 0

n = 555
print(reverse(n))

print("generic sum")
while n > 9:
    sum = 0
    while n > 0:
        sum += (n % 10)
        n //= 10
    n = sum

print(sum)

n = 153
print("Armstrong number")
lenn = int(math.log10(n))+1
arm, num = 0, n
while n > 0:
    arm += (n % 10) ** lenn
    n //= 10
print(arm)
print(arm == num)

print("fact")
n, f = 55, 1
for i in range(1, n+1):
    f *= i
print(f)

print("fib")
a = [0, 1]
for i in range(n):
    a.append(a[-1] + a[-2])
print(a)

print("n-th fib")
print(a[5])

print("strong number, perfect number")
print()