#!usr/bin/env
#echo "NUMBER OF CORES : 1" >> results.txt
#python3 main.py --num-processes=1 --epochs=2000 >> results.txt
echo "NUMBER OF CORES : 2" >> results.txt
python3 main.py --num-processes=2 --epochs=1212 >> results.txt
echo "NUMBER OF CORES : 3" >> results.txt
python3 main.py --num-processes=3 --epochs=1212 >> results.txt
echo "NUMBER OF CORES : 4" >> results.txt
python3 main.py --num-processes=4 --epochs=1212 >> results.txt
