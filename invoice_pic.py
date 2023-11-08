import os
from datetime import datetime
from PIL import Image
import re
from io import BytesIO
from cnocr import CnOcr
import numpy as np
import cv2
from invoice_qrcode import *
#import pdb

#部份源代码来自：https://blog.csdn.net/egowell/article/details/126626760
#后续生成票据图像时的大小，按照标准增值税发票版式240mmX140mm来设定
height_resize = 1400
width_resize = 2400

# 实例化不同用途CnOcr对象
ocr = CnOcr(name='',det_more_configs={'use_angle_clf': True}) #混合字符
ocr_numbers = CnOcr(name='number', cand_alphabet='0123456789.') #纯数字
ocr_UpperSerial = CnOcr(name='UpperSerial', cand_alphabet='0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ') #编号，只包括大写字母(没有O)与数字

def pic_decode(pic):
	"""
	# 去掉红色（印章）
	image = BytesIO(pic)
	buffer = BytesIO()
	image.save(buffer, format="JPEG")
	pic = buffer.getvalue()
	
	# 截取图片中部分区域图像 ：{名称: data: [坐标x, y, w, h ]，使用ocr的类型，0：混合字符，1：纯数字，2：编号，3：多行字符}
	crop_range_list = {
							'发票类型' : { 'data':[1000,0, 800, 200], 'type': 0},
							'发票代码' : { 'data':[1660, 70, 800, 55], 'type':1},
							'发票号码' : {'data':[1660, 115, 700,55], 'type': 1},
							'开票日期': {'data':[1660, 170, 800, 55], 'type': 0},
							'校验码' : {'data':[1660, 220, 900, 55], 'type':1}, 
							'项目名称' : {'data':[10, 600, 600, 450], 'type' : 3},
							'销售方名称' : {'data':[423, 1170, 933, 55], 'type': 0}, 
							'销售方纳税人识别号' : {'data':[423, 1225, 933, 55], 'type':2},
							'销售方地址电话' : {'data':[423, 1280, 933, 55], 'type': 0}, 
							'销售方开户行及账号' : {'data':[423, 1330, 933, 55], 'type': 0}, 
							'小计' : {'data':[1730, 1080, 450, 70], 'type': 0},
							'备注' : {'data':[1420, 1145, 900, 100], 'type': 3},
							}
	"""
	# 预处理图像
	warped = imagePreProcessing(pic)

	# 展示与保存预处理的图片---测试用
	#cv2.imshow('warpImage', warped)
	#cv2.imwrite('/root/result.jpg',warped)

	# 先模拟全图扫描一次
	result_list = ocr.ocr(warped)
	result_text = "\n".join([d['text'] for d in result_list])
	#print("识别结果：%s" % result_text)
	fapiaolexing = re_text(re.compile(r'.*(专用发票|普通发票).*'), result_text)
	fapiaodaima = re_text(re.compile(r'(发票代码|票据代码)(.*\d+)'), result_text)[5:]
	if not fapiaodaima.strip() :
		try:
			position = [item['position'] for item in result_list if item['text'][:4] == '发票代码'][0]
			x, y , w, h = int(position[0][0]), int(position[0][1]) - 5, 800, 55
			warped = cv2.rectangle(warped, (x, y), (x + w, y + h), (0, 255, 0), 2)
			#print("发票代码：%s, %s, %s, %s" %(x,y,w,h))
			fapiaodaima = cropOCR(cropImage(warped, [x, y, w, h]), 1).replace('o','0')
		except :
			print("没有找到：发票代码")
	fapiaohaoma = re_text(re.compile(r'(发票号码|票据号码)(.*\d+)'), result_text)[5:]
	if not fapiaohaoma.strip() :
		try:
			position = [item['position'] for item in result_list if item['text'][:4] == '发票号码'][0]
			x, y , w, h = int(position[0][0]), int(position[0][1]) - 5, 800, 55 
			warped = cv2.rectangle(warped, (x, y), (x + w, y + h), (0, 255, 0), 2)
			fapiaohaoma = cropOCR(cropImage(warped, [x, y, w, h]), 1).replace('o','0')
		except :
			print("没有找到：发票号码")
	kaipiaoriqi = re_text(re.compile(r'开票日期(.*)'), result_text)[5:]
	if not kaipiaoriqi.strip() :
		try:
			position = [item['position'] for item in result_list if item['text'][:4] == '开票日期'][0]
			x, y , w, h = int(position[0][0]), int(position[0][1]) - 5, 800, 55 
			warped = cv2.rectangle(warped, (x, y), (x + w, y + h), (0, 255, 0), 2)
			kaipiaoriqi = cropOCR(cropImage(warped, [x, y, w, h]), 0).replace('o','0')
		except :
			print("没有找到：开票日期")
	jiaoyan = re_text(re.compile(r'校验码\s*[:：]\s*([a-zA-Z0-9 ]+)'), result_text)[4:]
	if not jiaoyan.strip() :
		try:
			position = [item['position'] for item in result_list if item['text'][:3] == '校验码'][0]
			x, y , w, h = int(position[0][0]), int(position[0][1]) - 5, 900, 55 
			warped = cv2.rectangle(warped, (x, y), (x + w, y + h), (0, 255, 0), 2)
			jiaoyan = cropOCR(cropImage(warped, [x, y, w, h]), 1).replace('o','0')
		except :
			print("没有找到：校验码")
	xiaoxie = re.sub(r'[^\d.]','', (re_text(re.compile(r'小写.*(.*[0-9.]+)'), result_text)))
	if not xiaoxie :
		try :
			position = [item['position'] for item in result_list if item['text'][1:3] == '小写'][0]
			x, y , w, h = int(position[0][0]), int(position[0][1]) - 5, 450, 70 
			warped = cv2.rectangle(warped, (x, y), (x + w, y + h), (0, 255, 0), 2)
			xiaoxie = re.split("¥|￥",cropOCR(cropImage(warped, [x, y, w, h]), 1).replace('o','0'))[1]
		except :
			print("没有找到：小写")
	heji = ",".join(re.split("¥|￥",re_text(re.compile(r'合\s+计(.*)'), result_text))[1:])
	if len(heji) < 2 :
		try:
			position = [item['position'] for item in result_list if item['text'] == '合计'][0]
			x, y , w, h = int(position[0][0]), int(position[0][1]) - 5, 2400, 70 
			warped = cv2.rectangle(warped, (x, y), (x + w, y + h), (0, 255, 0), 2)
			heji = re.split("¥|￥",cropOCR(cropImage(warped, [x, y, w, h]), 0).replace('o','0'))[1]
			print("合计 %s" % heji)
			heji = heji[0]
			heji = heji[1]
		except :
			print("没有找到：合计")
			heji = ""
			tax = ""
		
	company = re.findall(re.compile(r'.*称\s*[:：]\s*([\u4e00-\u9fa5]+)'), result_text)
	if company:
		#print(re_block(company[len(company)-1]))
		gongsi = re_block(company[len(company)-1])
	else :
		gongsi = ""
	xixiang = re.findall(re.compile(r'([^\d\W]\w*)\*([^\d\W]\w*)'), result_text)
	if xixiang :
		xiangmu = ",".join([re_block(item) for sub in xixiang for item in sub])
	else :
		xiangmu = ""
	lst = {
		"类型":fapiaolexing, 
		"代码":fapiaodaima,
		"号码":fapiaohaoma,
		"开票日期":kaipiaoriqi,
		"校验码":jiaoyan,
		"小写":xiaoxie,
		"合计":heji,
		"税金":tax,
		"公司":gongsi, 
		"项目":xiangmu
	}
	if fapiaolexing == "" or fapiaodaima == "" or fapiaohaoma == "" or kaipiaoriqi == "" or jiaoyan == "" or heji == "" :
		#上面内容没有识别出来的，尝试用QRCODE方法再识别一次
		qrcode_result = decode_qrcode(Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))).split(',')
		if len(qrcode_result) >= 8 :
			# 增值税二维码格式说明：https://zhuanlan.zhihu.com/p/633025591?utm_id=0
			if qrcode_result[0] == '01' :
				if qrcode_result[1] == '01' :
					lst['类型'] = '增值税专用发票'
				elif qrcode_result[1] == '02' :
					lst['类型'] = '增值税普通发票'
				elif qrcode_result[1] == '08' :
					lst['类型'] = '增值税专用发票（电子）'
				elif qrcode_result[1] == '10' :
					lst['类型'] = '增值税普通发票（电子）'
				elif qrcode_result[1] == "32" :
					lst['类型'] = '（电子发票）普通发票'
					lst['小写'] = qrcode_result[4]
				lst['代码'] = qrcode_result[2]
				lst['号码'] = qrcode_result[3]
				if lst['类型'][:3] == "增值税" :
					lst['合计'] = qrcode_result[4]
				lst['开票日期'] = qrcode_result[5][0:4] + "年" + qrcode_result[5][4:6] + "月" + qrcode_result[5][6:8] + "日"
				lst['校验码'] = qrcode_result[6]
	if lst['税金'] == "" and lst['小写'] != "" and lst['合计'] != "" :
		lst['税金'] = str(float(lst['小写'])- float(lst['合计']))
	elif lst['合计'] == "" and lst['小写'] != "" and lst['税金'] != "" :
		lst['合计'] = str(float(lst['小写'])- float(lst['税金']))
	elif lst['小写'] == "" and lst['合计'] != "" and lst['税金'] != "" :
		lst['小写'] = str(float(lst['合计'])+ float(lst['税金']))
	print("解码结果：%s" % lst)

	"""
	# 处理预处理图像并将结果保存到text_ocr列表中
	text_ocr = {}
	
	for i in crop_range_list:
		#print(i, crop_range_list[i]['data'], crop_range_list[i]['type'])
		
		x, y, w, h = crop_range_list[i]['data'][0], crop_range_list[i]['data'][1], crop_range_list[i]['data'][2], crop_range_list[i]['data'][3]

		crop = cropImage(warped, [x, y, w, h])
		crop_text = cropOCR(crop, crop_range_list[i]['type'])
		crop_text = crop_text.replace('o','0') #发票中不会有小写字母o，凡是出现o的都使用0替代
		
		# 在图像上绘制矩形
		warped = cv2.rectangle(warped, (x, y), (x + w, y + h), (0, 255, 0), 2)

		text_str = str(crop_range_list[i]['data']) 
		# 在图片上绘制文字
		warped = cv2.putText(warped, text_str, (x, y + int(h/2.)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) , 2)
		
		print(i,':',crop_text)
	"""
	cv2.imwrite(os.path.join('/root', 'result.jpg'),warped)
	return lst

# 调整原始图片尺寸
def resizeImg(image, height=height_resize):
	h, w = image.shape[:2]
	pro = height / h
	size = (int(w * pro), int(height))
	img = cv2.resize(image, size)
	return img

# 边缘检测
def getCanny(image):
	# 高斯模糊
	binary = cv2.GaussianBlur(image, (3, 3), 2, 2)
	# 边缘检测
	binary = cv2.Canny(binary, 60, 240, apertureSize=3)
	# 膨胀操作，尽量使边缘闭合
	kernel = np.ones((3, 3), np.uint8)
	binary = cv2.dilate(binary, kernel, iterations=1)
	return binary

# 求出面积最大的轮廓
def findMaxContour(image):
	# 寻找边缘
	#contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	#contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
	contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	#contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# 计算面积
	max_area = 0.0
	max_contour = []
	for contour in contours:
		
		currentArea = cv2.contourArea(contour)
		if currentArea > max_area:
			max_area = currentArea
			max_contour = contour

	return max_contour, max_area

# 多边形拟合凸包的四个顶点
def getBoxPoint(contour):
	# 多边形拟合凸包，找四边形
	#hull = cv2.convexHull(contour)			# 
	epsilon = 0.05 * cv2.arcLength(contour, True)		# 计算整个凸包周长
	approx = cv2.approxPolyDP(contour, epsilon, True)		# cv2.approxPolyDP函数实现轮廓线的多边形逼近
	approx = approx.reshape((len(approx), 2))		#
	return approx

# 适配原四边形点集
def adapPoint(box, pro):
	box_pro = box
	if pro != 1.0:
		box_pro = box/pro
		box_pro = np.trunc(box_pro)
	return box_pro

# 四边形顶点排序，[top-left, top-right, bottom-right, bottom-left]
def orderPoints(pts):
	rect = np.zeros((4, 2), dtype='float32')
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect

# 增加发票头的高度
def addInvoiceHead(box):
	box[0] = find_C(box[3], box[0])
	box[1] = find_C(box[2], box[1])
	return box

	# 求C点座标
def find_C(A, B):
	diff = 0.3
	AB = np.array(B) - np.array(A)
	AC = AB * diff
	C = B + AC
	return C

# 计算长宽
def pointDistance(a, b):
	return int(np.sqrt(np.sum(np.square(a - b))))

# 透视变换
def warpImage(image, box):
	w, h = pointDistance(box[0], box[1]), pointDistance(box[1], box[2])
	dst_rect = np.array([[0, 0],
	[w - 1, 0],
	[w - 1, h - 1],
	[0, h - 1]], dtype='float32')
	#print("透视变换：%s, %s" %(box, dst_rect))
	M = cv2.getPerspectiveTransform(box, dst_rect)
	# 绘制矩形边框
	warped = cv2.warpPerspective(image, M, (w, h))
	return warped


# 统合图片预处理
def imagePreProcessing(path):
	# 下面两步是将内存中的图像转为cv2格式
	nparr = np.frombuffer(path, np.uint8)
	color_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

	# 转灰度、降噪
	image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
	#image = cv2.GaussianBlur(image, (3,3), 0)
	# 
	#_, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
	
	# 边缘检测、寻找轮廓、确定顶点
	ratio = height_resize / image.shape[0]
	img = resizeImg(image)
	binary_img = getCanny(img)
	cv2.imwrite(os.path.join('/root',"binary_img.jpg"), binary_img)
	
	max_contour, max_area = findMaxContour(binary_img)		# 找出最大的框
	
	temp_image=cv2.drawContours(img, max_contour, -1, (180, 180, 180), 3)
	cv2.imwrite(os.path.join('/root', 'result_1.jpg'),temp_image)
	
	boxes = getBoxPoint(max_contour)	# 多边形拟合凸包的四个顶点
	
	temp_image = cv2.polylines(img, [np.array(boxes, dtype='int32')], isClosed=True, color=(180, 180, 180), thickness=2)
	cv2.imwrite(os.path.join('/root', 'result_2.jpg'),temp_image)
	
	boxes = adapPoint(boxes, ratio)		# 适配原四边形点集
	temp_image = cv2.polylines(image, [np.array(boxes, dtype='int32')], isClosed=True, color=(180, 180, 180), thickness=2)
	cv2.imwrite(os.path.join('/root', 'result_3.jpg'),temp_image)
	
	boxes = orderPoints(boxes)		# 四边形顶点排序，[top-left, top-right, bottom-right, bottom-left]
	temp_image = cv2.polylines(image, [np.array(boxes, dtype='int32')], isClosed=True, color=(180, 180, 180), thickness=2)
	cv2.imwrite(os.path.join('/root', 'result_4.jpg'),temp_image)
	boxes = addInvoiceHead(boxes)		#移动上边界以获取发票头
	temp_image = cv2.polylines(image, [np.array(boxes, dtype='int32')], isClosed=True, color=(180, 180, 180), thickness=2)
	cv2.imwrite(os.path.join('/root', 'result_5.jpg'),temp_image)
	#print("裁剪尺寸：%s" % boxes)
	warped = warpImage(image, boxes)		#透视变换并裁剪图像

	# 调整最终图片大小
	height, width = warped.shape[:2]
	#size = (int(width*height_resize/height), height_resize)
	size = (width_resize, height_resize)
	warped = cv2.resize(warped, size, interpolation=cv2.INTER_CUBIC)

	cv2.imwrite(os.path.join('/root',"warp.jpg"), warped)
	
	return warped

# 截取图片中部分区域图像
def cropImage(img, crop_range):
	xpos, ypos, width, height = crop_range
	crop = img[ypos:ypos+height, xpos:xpos+width]
	return crop

# 从截取图片中识别文字
def cropOCR(crop, ocrType):
	if ocrType==0:
		text_crop_list = ocr.ocr_for_single_line(crop)
		text_crop = ''.join(text_crop_list['text'])
	elif ocrType==1:
		text_crop_list = ocr_numbers.ocr_for_single_line(crop)
		text_crop = ''.join(text_crop_list['text'])
	elif ocrType==2:
		text_crop_list = ocr_UpperSerial.ocr_for_single_line(crop)
		text_crop = ''.join(text_crop_list['text'])
	elif ocrType==3:
		text_crop_list = ocr.ocr(crop)
		text_crop = ""
		for i in text_crop_list:
			text_crop = text_crop + ',' + i['text']
		if text_crop[:1] == "," :
			text_crop = text_crop[1:]
	#print("识别结果:%s" % text_crop_list)
	return text_crop

def re_text(bt, text):
	m1 = re.search(bt, text)
	if m1 is not None:
		return re_block(m1[0])
	else :
		return ""

def re_block(text):
	return text.replace(' ', '').replace('　', '').replace('）', '').replace(')', '').replace('：', ':')


"""
# 尝试用其它方法变型，目前失败
def new_warp(image):
	
	def find_lines(image):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(gray, 50, 150, apertureSize=3)
		lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
		return lines

	def find_quadrilateral(lines):
		left_lines = []
		right_lines = []
		top_lines = []
		bottom_lines = []

		for line in lines:
			x1, y1, x2, y2 = line[0]
			slope = (y2 - y1) / (x2 - x1)

			if x1 < image.shape[1] // 2 and x2 < image.shape[1] // 2:
				left_lines.append((x1, y1, x2, y2))
			elif x1 >= image.shape[1] // 2 and x2 >= image.shape[1] // 2:
				right_lines.append((x1, y1, x2, y2))

			if y1 < image.shape[0] // 2 and y2 < image.shape[0] // 2:
				top_lines.append((x1, y1, x2, y2))
			elif y1 >= image.shape[0] // 2 and y2 >= image.shape[0] // 2:
				bottom_lines.append((x1, y1, x2, y2))

		left_line = min(left_lines, key=lambda x: x[0])
		right_line = max(right_lines, key=lambda x: x[0])
		top_line = min(top_lines, key=lambda x: x[1])
		bottom_line = max(bottom_lines, key=lambda x: x[1])

		return [left_line, right_line, top_line, bottom_line]

	def perspective_transform(image, lines):
		
		#[[         16         467          16         116]
 #[        776         467         776         114]
 #[        561          28         681          28]
 #[        583         479         769         479]]
		
		
		src_points = addInvoiceHead(np.array([[lines[0][2], lines[0][3]], [lines[1][2], lines[1][3]], [lines[1][0], lines[1][1]], [lines[0][0], lines[0][1]]], dtype='float32'))
		dst_points = np.array([[0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]],dtype='float32')
		print("变型座标：%s, %s" %(src_points,dst_points))
		matrix = cv2.getPerspectiveTransform(src_points, dst_points)
		transformed_image = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))

		return transformed_image

	def crop_quadrilateral(image, lines):
		quadrilateral_points = np.float32([lines[0], lines[1], lines[2], lines[3]])
		width = int(np.sqrt((quadrilateral_points[1][0] - quadrilateral_points[0][0]) ** 2 + (quadrilateral_points[1][1] - quadrilateral_points[0][1]) ** 2))
		height = int(np.sqrt((quadrilateral_points[2][0] - quadrilateral_points[1][0]) ** 2 + (quadrilateral_points[2][1] - quadrilateral_points[1][1]) ** 2))

		x = int(quadrilateral_points[0][0] - width // 2)
		y = int(quadrilateral_points[0][1] - height // 2)

		cropped_image = image[y:y+height, x:x+width]

		return cropped_image

	
	lines = find_lines(image)
	print("直线：%s" % lines)
	quadrilateral_lines = find_quadrilateral(lines)
	transformed_image = perspective_transform(image, quadrilateral_lines)
	cv2.imwrite(os.path.join('/root',"transformed_image.jpg"), transformed_image)
	#cropped_image = crop_quadrilateral(transformed_image, quadrilateral_lines)
	return transformed_image
"""
	

