#include "stdio.h"
int main(int argc, char *argv[]){
  if(argc != 2){
    printf("wrong arguments");
  }else{
    char * filename = argv[1];
    printf("%s\n", filename);
  }
}
