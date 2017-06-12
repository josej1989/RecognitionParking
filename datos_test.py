# Procesamiento de datos de prueba

def process_test_data():
	testing_data = []
	for img in tqdm(os.listdir(DIR_TEST)):
		path = os.path.join(DIR_TEST)
		img_num = img.split('.')[0]
		img = cv2.imread(path+img,cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img, (tamano,tamano))
		testing_data.append([np.array(img), img_num])
	np.save('test_data.npy', testing_data)
	return testing_data