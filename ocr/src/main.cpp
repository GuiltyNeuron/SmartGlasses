// Project : NLP ++
// Author : Achraf KHAZRI AI Research Engineer
// Script : Livenstein distance

#include <iostream>
#include <string>
#include <algorithm>

using namespace std;

int LD(string term1, string term2)
{
	int out;
    int cost;

    if (term1.length() == 0) out = term2.length();
    if (term2.length() == 0) out = term1.length();

    //if( term1.substr(-1,0) == term2.substr(-1,0)) cost = 0;
    //else cost = 1;

    // int res = min(LD(term1.substr(0,term1.length() - 1), term2) + 1, LD(term1, term2.substr(0,term2.length() - 1)) + 1, LD(term1.substr(0,term1.length() - 1), term2.substr(0,term2.length() - 1)) + cost);
	//int out = min(2, 5);
    cout << "hello world" <<endl;
    
	return out;


}

void main( )
{
   int x = LD("Hi","Achraf");
}

