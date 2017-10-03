#make an ascii digital clock for terminal cron

#figure out how to display numbers
import time

hour = int(time.strftime("%H", time.localtime()))
minute = int(time.strftime("%M", time.localtime()))

charlist = []
#12 hour time, this isn't europe
if hour > 12:
	hour-=12

empty = 6*[5*[0]]

text1 =['   1',
        '   1',
        '   1',
        '   1',
        '   1',
        '   1']


 
text2 = [' 222 ',
 	 '2   2',
	 '    2',
	 '  2  ',
	 ' 2   ',
	 ' 2222']
text3 = ['3333 ','    3',' 333 ','    3','    3','3333 ']

text4 = ['4   4',
	 '4   4',
	 '44444',
	 '    4',
	 '    4',
	 '    4']

text5 = ['55555',
	 '5    ',
	 '5555 ',
	 '    5',
	 '    5',
	 '5555 ']

text6 = [' 666 ', 
         '6    ',
         '6666 ',
         '6   6',
         '6   6',
         ' 666 ']

text7 = ['77777',
	 '    7', 
	 '   7 ',
	 '  7  ',
	 ' 7   ',
	 ' 7   ']

text8 = [' 888 ',
	 '8   8',
	 ' 888 ',
	 '8   8',
	 '8   8',
	 ' 888 ']

text9 = ['9999 ',
         '9   9',
         '99999',
         '    9',
         '    9',
         '   9 ']

text0 = [' 000 ',
	 '0   0',
	 '0   0',
	 '0   0',
	 '0   0',
	 ' 000 ']

text = [text0, text1, text2, text3, text4, text5, text6, text7, text8, text9]

clockface = []

#2 digit check
if hour >= 10:
	clockface.append(text[int(str(hour)[0])])
	clockface.append(text[int(str(hour)[1])])
else:
	clockface.append(text[0])	
	clockface.append(text[hour])

if minute < 10:
	clockface.append(text[0])
	clockface.append(text[int(str(minute)[0])])
else:
	clockface.append(text[int(str(minute)[0])])
	clockface.append(text[int(str(minute)[1])])


def arraytotext(array):
	joinedarray = 6*['']
	for line in range(6):
		for numeral in range(4):
			if numeral != 2:
				joinedarray[line]+='  ' + array[numeral][line]
			elif line == 2 or line ==4:	
				joinedarray[line]+=' * ' + array[numeral][line]
			else:	
				joinedarray[line]+='   ' + array[numeral][line]
	#turn array into string separated by \n
	output = ''
	for row in joinedarray:
		output += "\n" + row
	return output

print('\n'+arraytotext(clockface)+'\n')
''
