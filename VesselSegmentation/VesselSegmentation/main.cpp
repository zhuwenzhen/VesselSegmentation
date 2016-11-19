//
//  main.cpp
//  VesselSegmentation
//
//  Created by shislis on 11/19/16.
//  Copyright © 2016 shislis. All rights reserved.
//

#include <iostream>

int main(int argc, const char * argv[]) {
    // insert code here...
    if(argc != 2){
        printf("wrong arguments\n");
    }else{
        const char * filename = argv[1];
        printf("%s\n", filename);
    }
    return 0;
}
