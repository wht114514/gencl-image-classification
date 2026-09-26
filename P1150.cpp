#include <iostream>
using namespace std;
int main()
{
	double n,k,a;
	cin >> n >> k;
	a=n;
	while (n>=k)
	{
		a+=n/k;
		n=n/k;
	 } 
	cout << (int)a;
	return 0;
}
