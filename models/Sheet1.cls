VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "Sheet1"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = True
Attribute VB_Exposed = True
Private Sub CommandButton1_Click()
a = InputBox("Please Enter Structural Number")
MsgBox (a)

Worksheets("Sheet1").Cells(2, 2).Value = a
 
b = InputBox("Please Enter AEDT")
MsgBox (b)

Worksheets("Sheet1").Cells(3, 2).Value = b

 End Sub

