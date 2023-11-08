from invoice_pic import *
from invoice_pdf import *

# example for invoice file decode
a="/root/test_invoice.pdf"
print(pdf_decode(open(a,'rb').read()))
b="/root/test_invoice.jpg"
print(pic_decode(open(b,'rb').read()))
