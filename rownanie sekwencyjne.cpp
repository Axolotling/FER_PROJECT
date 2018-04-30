#include <iostream>
#include <fstream>
#include <queue>
 
using namespace std;
 
queue <string> MQ[7];
 
int main()
{
    string linia;
    ofstream plik2;
    ifstream plik;
	plik.open("dataset.csv");
	plik2.open("dataset2.csv");

    while(getline(plik, linia))
       	MQ[linia[0]-'0'].push(linia);
		
	bool empty=false;
	int k=0;
	while(!empty)
		{
		k++;
		for(int i=0;i<7;i++)
			{
			plik2<<MQ[i].front()<<endl;
			MQ[i].pop();
			if(MQ[i].empty())
			empty=true;
			}
		}
		
	cout<<k<<endl;
}
