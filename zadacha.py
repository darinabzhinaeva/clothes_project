n=int(input())
i=2
is_prime=True
while i < n:
    if n % i==0:
        is_prime=False
        break
    i+=1

if is_prime:
    print('prime')
else:
    print('composite')