# import module 

import calendar
#print(dir(calendar))

print(calendar.prmonth(2020, 9))

yy = 2019
mm = 9
     
print(calendar.month(yy, mm))

print("-" * 20)

print ("The calender of year 2020 is : ")  
print (calendar.calendar(2020))


"""
w = The width between two columns. Default value is 2. 
l = Blank line between two rows. Default value is 1.
c = Space between two months (Column wise). Default value is 6.
m = Number of months in a row. Default value is 3.

"""
print ("The calender of year 2021 is : ")  
print (calendar.calendar(2021, w=2, l=1, c=6, m=4))


