normal_conditions = """
Normal_condition1:There are two [terminal]s with a width of 500 or less, and their height difference is 50 or less.
Function1:

Normal_condition2:There is one [cable] with a height of 150 or less, and its width is 600 or more.
Function2:

Normal_condition3:Calculate the height of the cable, cable_height, where the height is 150 or less and the width is 600 or more, and calculate the difference between the top edges of terminals, terminal_diff, where the width is 500 or less. The absolute value of the difference between cable_height and terminal_diff (|cable_height - terminal_diff|) must be smaller than 50.
Function3:

Normal_condition4:If the color of the cable is yellow, the height of one [terminal] with a width of less than 500 is 178-238.
Function4:

Normal_condition5:If the color of the cable is blue, the height of one [terminal] with a width of less than 500 is 260-311.
Function5:

Normal_condition6:If the color of the cable is red, the height of one [terminal] with a width of less than 500 is 431-467.
Function6:

Normal_condition7:Detect SA. Category is "splicing_connectors". Components is {'cable':1}
Function7:

Create 7 python function.
"""
