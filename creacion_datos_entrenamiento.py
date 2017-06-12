# Crear datos de entrenamiento de la red neuronal

def create_train_data():
	training_data = []
	for img in tqdm(os.listdir(DIR_ENTRENADOR)):
		label = label_img(img)
		ruta = os.path.join(DIR_ENTRENADOR)
		img = cv2.imread(ruta+img, cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img, (tamano,tamano))
		training_data.append([np.array(img), np.array(label)])
	shuffle(training_data)
	np.save('train_data.npy', training_data)
	return training_data