import org.apache.spark.sql.SparkSession
import spark.implicits._
import org.apache.spark.sql.Column
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

//Iniciamos sesion en spark para la creacion del Schema
val spark = SparkSession.builder.master("local[*]").getOrCreate()
//Se crea un dataframe para la lectura de los datos a limpiar con el archivo iris.csv
val df = spark.read.option("inferSchema","true").csv("Iris.csv").toDF(
  "SepalLength", "SepalWidth", "PetalLength", "PetalWidth","class"
)
//Se crea la nueva columna con los datos a limpiar para la lectura del algoritmo de machine learning
val newcol = when($"class".contains("Iris-setosa"), 1.0).
  otherwise(when($"class".contains("Iris-virginica"), 3.0).
  otherwise(2.0))
val newdf = df.withColumn("etiqueta", newcol)
newdf.select(
  "etiqueta",
  "SepalLength",
   "SepalWidth",
    "PetalLength",
     "PetalWidth",
     "class").show(150, false)

//Se juntan los datos con assembler.
val assembler = new VectorAssembler()  .setInputCols(Array("SepalLength",
   "SepalWidth",
    "PetalLength",
     "PetalWidth",
     "etiqueta")).setOutputCol("features")
//Siguiente empieza la transformacion de la informacion con las columnas
val features = assembler.transform(newdf)
features.show(5)

// Se indexan los labels para añadirlos al metadata para incluirlos todos al index
val labelIndexer = new StringIndexer().setInputCol("class").setOutputCol("indexedLabel").fit(features)
println(s"Found labels: ${labelIndexer.labels.mkString("[", ", ", "]")}")

// Se categorizan las caracteristicas (Features) y los indexa
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures")
.setMaxCategories(4).fit(features)

//Se crean las Variables de entrenamiento, de text y las variables al azar.
val splits = features.randomSplit(Array(0.6, 0.4))
val trainingData = splits(0)
val testData = splits(1)

// Se muestra la estructura para la red neuronal
val layers = Array[Int](5, 5, 5, 3)

// crea la variable entrenador y asignamos los parametros
val trainer = new MultilayerPerceptronClassifier().setLayers(layers)
.setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
.setBlockSize(128).setSeed(System.currentTimeMillis).setMaxIter(200)

// Reconvierte los labels indexados a su forma original
val labelConverter = new IndexToString().setInputCol("prediction")
.setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

// Se Encadena los indexados y la  MultilayerPerceptronClassifier en una  Pipeline.
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, trainer, labelConverter))

//El modelo entrena los datos indexados del entrenador
val model = pipeline.fit(trainingData)

//Trabaja haciendo las predicciones del modelo
val predictions = model.transform(testData)
predictions.show(5)

// Arroja el resultado de la prediccion, el label original y plasma el error de test.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel")
.setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

//Explique detalladamente el procesos Perceptron multilayer.
/*El modelo de Perceptron multilayer es una red neuronal artificial formada por diferentes capas que
constan de dos aprendizajes para resolver procesos matematicos con logica definida en las variables.*/

/*La funcion de multilayer Perceptron utiliza funciones de nodos ocultos y nodos de salida que se basan en la cantidad de entradas en la
red neuronal, junto con los nodos ocultos (logistica) y resultados de salida con la funcion softmax quye devuelve la precision del modelo
y la diferencia (error) con las salidas
/*

//Examen Unidad 2 Flores Reyes Jose Luis 14212327//
