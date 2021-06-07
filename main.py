import sys
import Enhance

def help():
    print(
    """To use this python code, follow this :
    python3 main.py -r input_path [-e ouputname]
    -r : define the input path, ex. test.jpg
    -e : define the output name, ex. testUpscaled.jpg (optional)
    """)

if len(sys.argv)==3 and sys.argv[1]=='-r':
    name_ouput = sys.argv[2].split("/")[-1].split('.')[0]+'UpScaled.jpg'
    Enhance.Enhance(sys.argv[2],name_ouput)

elif len(sys.argv)==5 and sys.argv[1]=='-r' and sys.argv[3]=='-e':
    Enhance.Enhance(sys.argv[2],sys.argv[4])

else:
    help()