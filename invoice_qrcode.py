from pyzbar.pyzbar import decode as qrdecode

def decode_qrcode(image):
#	img = Image.open(image_path)
	decoded_objects = qrdecode(image)
	result = ""
	for obj in decoded_objects:
#		print("类型：", obj.type)
#		print("数据：", obj.data.decode("utf-8"))
		if obj.type == "QRCODE" :
			result = obj.data.decode('utf-8')
			print("QRCODE结果: %s" % result)
			break
	return result
