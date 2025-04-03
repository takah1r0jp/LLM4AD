normal_conditions = """
Normal_condition1:There is a [label with a fruit illustration].  
Function1:

Normal_condition2:There is [a label with text] written on it.
Function2:

Normal_condition3:The [label with a fruit illustration] is above the [label with text].
Function3:

Normal_condition4:The difference between the horizontal center of the [label with a fruit illustration] and the horizontal center of the [fruit illustration] with a height less than 150 is 20 or less. Do not use delete_overlaps() in this function.
Function4:

Normal_condition5:The difference between the vertical center of the [label with a fruit illustration] and the vertical center of the [fruit illustration] with a height less than 150 is 20 or less. Do not use delete_overlaps() in this function.
Function5:

Normal_condition6:There are [juice] in the image.
Function6:

Normal_condition7:The difference between the horizontal centers of the [label with a fruit illustration] and the [label with text] is within 30.
Function7:

Normal_condition8:The distance between the vertical centers of the [label with a fruit illustration] and the [label with text] is at least 370 and at most 500.
Function8:

Normal_condition9:If the [juice] is red, the [illustration on the label] is a cherry.
Function9:

Normal_condition10:If the [juice] is white, the [illustration on the label] is a banana.
Function10:

Normal_condition11:If the [juice] is orange, the [illustration on the label] is an orange.
Function11:

Create 12 python functions.
"""
