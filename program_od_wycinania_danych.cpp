#include <iostream>
#include <fstream>

 
using namespace std;
 
int main()
{
    string linia;
    ofstream plik2;
    ifstream plik;
	plik.open("fer2013.csv");
	plik2.open("dataset.csv");
 
    int i=0;
        while(getline(plik, linia))
        	{
        	i++;
            if(i>1)
            	{
            	int przecinek=0;
            	for(int k=0;k<linia.size();k++)
            		{
            		if(linia[k]==',')
            			przecinek++;
            		plik2<<linia[k];
            		if(przecinek>1)
            			{
            			plik2<<"\n";
            			break;
            			}
					}
				
				}
        }
}
