package algorithms;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.stream.Stream;

/**
 * @author soyeb84
 */

public class KNearestNeighbour {
	private final static String TRAINING_SET_FILE = "d:\\breast_cancer_train.csv";
	private final static String TEST_SET_FILE = "d:\\breast_cancer_test.csv";
	private static final int NORMALIZATION_MIN = 0;
	private static final int NORMALIZATION_MAX = 1;
	private static final int DEFAULT_K = 100;

	private static final Summary summary = new Summary();

	public static void main(String args[]) throws Exception {
		List<String> trainingSetData = readAllLines(TRAINING_SET_FILE);
		List<String> testDataSet = readAllLines(TEST_SET_FILE);
		List<PatientData> trainingSet = convertRowToPatientData(trainingSetData);
		int[] k = {1,3,5,7,9};
		for (int kValue : k) {
			List<PatientData> testSet = convertRowToPatientData(testDataSet);
			double truePositive=0;
			double trueNegative=0;
			double falsePositive=0;
			double falseNegative=0;
			System.out.println("------------------------------------------------");
			System.out.println(String.format("Computing for k=%s",kValue));
			try (PrintWriter pw = new PrintWriter(new File("d:\\predictions" + kValue + ".csv"))) {
				//run loop for each patient specified in test set.
				for (PatientData testPatient : testSet) {
					Set<PatientData> nearestNbours = findKNearestNeigbours(testPatient, trainingSet, kValue);
					TumorClass actualClass = testPatient.getTumorClass();
					/** Number of patients from neighbourhood with malignant Tumor.**/
					int malignantCount = 0;
					
					/** Number of patients from neighbourhood with benign Tumor **/
					int benignCount = 0;
					
					for (PatientData nbor : nearestNbours) {
//						System.out.println(String.format("neighbor id is: %s with class %s" , nbor.getPatientId(),nbor.getTumorClass()));
						if (nbor.getTumorClass() == TumorClass.MALIGN) {
							malignantCount++;
						} else {
							benignCount++;
						}
					}
					TumorClass assignedClass = null;
					
					//If the number of patients in the neighborhood with benign tumor is more then assign benign class.
					if (benignCount > malignantCount) {
						assignedClass = TumorClass.BENIGN;
					}else{
						assignedClass = TumorClass.MALIGN;
					}
					
					
					
					if(assignedClass==TumorClass.MALIGN){
						if(actualClass==assignedClass){
							//if actual and assigned are malignant.
							truePositive++;
						}else{
							//if actual is benign but assigned is malign.
							falsePositive++;
						}
					}else{
						if(actualClass==assignedClass){
							//if  actual and assigned class is benign
							trueNegative++;
						}else{
							// if assigned is benign but actual is malign.
							falseNegative++;
						}
					}
					testPatient.setTumorClass(assignedClass);
					pw.println(String.format("%s, %s", testPatient.getPatientId(),
							testPatient.getTumorClass().name().toLowerCase()));
				}
				
				/**
				 * Formulas derived from <a>https://en.wikipedia.org/wiki/Sensitivity_and_specificity</a>
				 */
				System.out.println("sensitivity % "+ (truePositive/(truePositive+falsePositive))*100);
				System.out.println("Specificity % "+(trueNegative/(trueNegative+falseNegative))*100);
				System.out.println(String.format("Accuracy is %s",((trueNegative + truePositive)/(testSet.size()))));
				System.out.println(String.format("Precision is %s",((truePositive)/(truePositive + falsePositive))));
				System.out.println(String.format("%40s", "Actual Value"));
				
				System.out.println(String.format("%31s|%s","Malign","Benign"));
				System.out.println(String.format("Predicted Value%10s|%5s|%5s","Malign",truePositive,falseNegative ));
				System.out.println(String.format("%25s|%5s|%5s","Benign",falsePositive,trueNegative ));
			}

		}

		
	}

	/**
	 * Finds k nearest nbors by calculating distance between the PatientData in
	 * question and trainging set.
	 * 
	 * @param patientData
	 * @param k
	 * @return
	 */
	private static Set<PatientData> findKNearestNeigbours(final PatientData patientDataToTest,
			List<PatientData> trainingSet, final int k) {
		Set<PatientData> nearestNbours = new HashSet<>();
		Map<DistanceWrapper, PatientData> distanceMap = new TreeMap<>();
		for (PatientData patientData : trainingSet) {
			double distance = getDistance(patientDataToTest, patientData);
			distanceMap.put(new DistanceWrapper(distance), patientData);

		}

		distanceMap.keySet().stream().sorted(new Comparator<DistanceWrapper>() {
			@Override
			public int compare(DistanceWrapper o1, DistanceWrapper o2) {
				return Double.compare(o1.distance, o2.distance);
			}
		}).skip(0).limit(k).forEach(x -> nearestNbours.add(distanceMap.get(x)));
		;

		return nearestNbours;

	}

	
	/**
	 * Reads all the lines from the file.
	 * @param absolutePath
	 * @return
	 * @throws IOException
	 */
	private static List<String> readAllLines(final String absolutePath) throws IOException {
		File f = new File(absolutePath);
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(f)));
		List<String> lines = new ArrayList<>();
		String line = null;

		while ((line = br.readLine()) != null) {
			if (line.startsWith("\"")) {
				continue;
			}
			if (!line.trim().isEmpty()) {
				lines.add(line);
			}
		}
		return lines;
	}

	/**
	 * ' Constucts a patientdata object from a row read from the string file.
	 * The conversion should take care of the attribute scaling/normalization.
	 * 
	 * @param row
	 *            a row from training/test dataset
	 * @return a PatientData object
	 */
	private static List<PatientData> convertRowToPatientData(final List<String> rows) {
		List<PatientData> patientDataList = new ArrayList<>();
		double[][] dataMatrix = new double[rows.size()][9];
		int i = 0;
		for (String row : rows) {
			String[] split = row.split(",");
			// skip first column as it only contains patient's ID.
			dataMatrix[i][0] = Double.valueOf(split[1]);
			dataMatrix[i][1] = Double.valueOf(split[2]);
			dataMatrix[i][2] = Double.valueOf(split[3]);
			dataMatrix[i][3] = Double.valueOf(split[4]);
			dataMatrix[i][4] = Double.valueOf(split[5]);
			dataMatrix[i][5] = Double.valueOf(split[6]);
			dataMatrix[i][6] = Double.valueOf(split[7]);
			dataMatrix[i][7] = Double.valueOf(split[8]);
			dataMatrix[i][8] = Double.valueOf(split[9]);
			i++;
		}
		//Extract column from matrix foreach field
		double[] clumpThickness = extractColumn(dataMatrix, 0);
		double[] cellSizes = extractColumn(dataMatrix, 1);
		double[] cellShapes = extractColumn(dataMatrix, 2);
		double[] markAdeisons = extractColumn(dataMatrix, 3);
		double[] epitCellSizes = extractColumn(dataMatrix, 4);
		double[] bareNuclei = extractColumn(dataMatrix, 5);
		double[] blandChromos = extractColumn(dataMatrix, 6);
		double[] nNucleois = extractColumn(dataMatrix, 7);
		double[] mitesosis = extractColumn(dataMatrix, 8);

		//calculates min and max value for each field that we will use for min-max normalization.
		summary.setClumpThicknessMax(Arrays.stream(clumpThickness).max().getAsDouble());
		summary.setClumpThicknessMin(Arrays.stream(clumpThickness).min().getAsDouble());
		summary.setCellSizeMax(Arrays.stream(cellSizes).max().getAsDouble());
		summary.setCellSizeMin(Arrays.stream(cellSizes).min().getAsDouble());
		summary.setCellShapesMax(Arrays.stream(cellShapes).max().getAsDouble());
		summary.setCellShapeMin(Arrays.stream(cellShapes).min().getAsDouble());
		summary.setMargAdeisonMax(Arrays.stream(markAdeisons).max().getAsDouble());
		summary.setMargAdeisonMin(Arrays.stream(markAdeisons).min().getAsDouble());
		summary.setEpitCellMax(Arrays.stream(epitCellSizes).max().getAsDouble());
		summary.setEpitCellMin(Arrays.stream(epitCellSizes).min().getAsDouble());
		summary.setBareNucleiMax(Arrays.stream(bareNuclei).max().getAsDouble());
		summary.setBareNucleiMin(Arrays.stream(bareNuclei).min().getAsDouble());
		summary.setBlandChromoMax(Arrays.stream(blandChromos).max().getAsDouble());
		summary.setBlandChromoMin(Arrays.stream(blandChromos).min().getAsDouble());
		summary.setnNucleoliMax(Arrays.stream(nNucleois).max().getAsDouble());
		summary.setnNucleoliMin(Arrays.stream(nNucleois).min().getAsDouble());
		summary.setMitosesMax(Arrays.stream(mitesosis).max().getAsDouble());
		summary.setMitosesMin(Arrays.stream(mitesosis).min().getAsDouble());
		
		//normalize the value of each field before storing it to the object.
		for (int j = 0; j < rows.size(); j++) {
			PatientData patientData = new PatientData();
			patientData.setBareNuclei(getNormalizedValue(dataMatrix[j][FieldName.BARE_NUCLEI.getIndex()],
					summary.getBareNucleiMin(), summary.getBareNucleiMax()));

			patientData.setBlandChromo(getNormalizedValue(dataMatrix[j][FieldName.BLAND_CHROMO.getIndex()],
					summary.getBlandChromoMin(), summary.getBlandChromoMax()));

			patientData.setCellShape(getNormalizedValue(dataMatrix[j][FieldName.CELL_SHAPE.getIndex()],
					summary.getCellShapeMin(), summary.getCellShapesMax()));

			patientData.setnNucleoli(getNormalizedValue(dataMatrix[j][FieldName.N_NUCLEI.getIndex()],
					summary.getnNucleoliMin(), summary.getnNucleoliMax()));

			patientData.setMitoses(getNormalizedValue(dataMatrix[j][FieldName.MITESOIS.getIndex()],
					summary.getMitosesMin(), summary.getMitosesMax()));

			patientData.setMargAdeison(getNormalizedValue(dataMatrix[j][FieldName.MARG_ADEISON.getIndex()],
					summary.getMargAdeisonMin(), summary.getMargAdeisonMax()));

			patientData.setCellSize(getNormalizedValue(dataMatrix[j][FieldName.CELL_SIZE.getIndex()],
					summary.getCellSizeMin(), summary.getCellSizeMax()));

			patientData.setEpitCellSize(getNormalizedValue(dataMatrix[j][FieldName.EPIT_CELL_SIZE.getIndex()],
					summary.getEpitCellMin(), summary.getEpitCellMax()));

			patientData.setClumpThickness(getNormalizedValue(dataMatrix[j][FieldName.CLUMP_THICKNESS.getIndex()],
					summary.getClumpThicknessMin(), summary.getClumpThicknessMax()));
			patientData.setTumorClass(TumorClass.fromName((rows.get(j).split(",")[10])));
			patientData.setPatientId(rows.get(j).split(",")[0]);

			patientDataList.add(patientData);
		}

		return patientDataList;
	}

	/**
	 * Rescales value using Min - Max normalization.
	 * @param value
	 * @param minValue
	 * @param maxValue
	 * @return
	 */
	private static double getNormalizedValue(double value, double minValue, double maxValue) {
		return (((value - minValue) / (maxValue - minValue)) * (NORMALIZATION_MAX - NORMALIZATION_MIN))
				+ NORMALIZATION_MIN;
		
	}

	/**
	 * Extracts a particular column from a 2D array.
	 * @param array
	 * @param col
	 * @return
	 */
	private static double[] extractColumn(double[][] array, int col) {
		double[] arrayToReturn = new double[array.length];
		for (int i = 0; i < array.length; i++) {
			arrayToReturn[i] = array[i][col];
		}
		return arrayToReturn;
	}

	/**
	 * Calculates Euclidean distance between two {@link PatientData}, which are
	 * essentially two rows in the dataset.
	 * 
	 * @param patientData1
	 *            observation of first patient.
	 * @param patientData2
	 *            observation of second patient.
	 * @return
	 */
	private static double getDistance(PatientData patientData1, PatientData patientData2) {
		double distance = Math.abs(Math.sqrt(Math.pow(patientData1.getBareNuclei() - patientData2.getBareNuclei(), 2)
				+ Math.pow(patientData1.getBlandChromo() - patientData2.getBlandChromo(), 2)
				+ Math.pow(patientData1.getCellShape() - patientData2.getCellShape(), 2)
				+ Math.pow(patientData1.getCellSize() - patientData2.getCellSize(), 2)
				+ Math.pow(patientData1.getClumpThickness() - patientData2.getClumpThickness(), 2)
				+ Math.pow(patientData1.getEpitCellSize() - patientData2.getEpitCellSize(), 2)
				+ Math.pow(patientData1.getMargAdeison() - patientData2.getMargAdeison(), 2)
				+ Math.pow(patientData1.getMitoses() - patientData2.getMitoses(), 2)
				+ Math.pow(patientData1.getnNucleoli() - patientData2.getnNucleoli(), 2)));

		// System.out.println(String.format("Distance between %s and %s is
		// %s",patientData1.getPatientId(),patientData2.getPatientId(),distance));
		return distance;

	}

}

enum FieldName {
	CLUMP_THICKNESS(0), CELL_SIZE(1), CELL_SHAPE(2), MARG_ADEISON(3), EPIT_CELL_SIZE(4), BARE_NUCLEI(5), BLAND_CHROMO(
			6), N_NUCLEI(7), MITESOIS(8);

	private int index;

	private FieldName(final int index) {
		this.index = index;
	}

	public int getIndex() {
		return index;
	}

}

class DistanceWrapper implements Comparable<DistanceWrapper> {
	double distance;

	public DistanceWrapper(double ditance) {
		this.distance = ditance;
	}

	@Override
	public int compareTo(DistanceWrapper arg0) {
		return Double.compare(this.distance, arg0.distance);
	}
}

class Summary {
	private double clumpThicknessMax;
	private double clumpThicknessMin;
	private double cellSizeMin;
	private double cellSizeMax;
	private double cellShapesMax;
	private double cellShapeMin;
	private double margAdeisonMin;
	private double margAdeisonMax;
	private double epitCellMin;
	private double epitCellMax;
	private double bareNucleiMin;
	private double bareNucleiMax;
	private double blandChromoMin;
	private double blandChromoMax;
	private double nNucleoliMin;
	private double nNucleoliMax;
	private double mitosesMax;
	private double mitosesMin;

	public double getCellShapesMax() {
		return cellShapesMax;
	}

	public void setCellShapesMax(double cellShapesMax) {
		this.cellShapesMax = cellShapesMax;
	}

	public double getCellShapeMin() {
		return cellShapeMin;
	}

	public void setCellShapeMin(double cellShapeMin) {
		this.cellShapeMin = cellShapeMin;
	}

	public double getClumpThicknessMax() {
		return clumpThicknessMax;
	}

	public void setClumpThicknessMax(double clumpThicknessMax) {
		this.clumpThicknessMax = clumpThicknessMax;
	}

	public double getClumpThicknessMin() {
		return clumpThicknessMin;
	}

	public void setClumpThicknessMin(double clumpThicknessMin) {
		this.clumpThicknessMin = clumpThicknessMin;
	}

	public double getCellSizeMin() {
		return cellSizeMin;
	}

	public void setCellSizeMin(double cellSizeMin) {
		this.cellSizeMin = cellSizeMin;
	}

	public double getCellSizeMax() {
		return cellSizeMax;
	}

	public void setCellSizeMax(double cellSizeMax) {
		this.cellSizeMax = cellSizeMax;
	}

	public double getMargAdeisonMin() {
		return margAdeisonMin;
	}

	public void setMargAdeisonMin(double margAdeisonMin) {
		this.margAdeisonMin = margAdeisonMin;
	}

	public double getMargAdeisonMax() {
		return margAdeisonMax;
	}

	public void setMargAdeisonMax(double margAdeisonMax) {
		this.margAdeisonMax = margAdeisonMax;
	}

	public double getEpitCellMin() {
		return epitCellMin;
	}

	public void setEpitCellMin(double epitCellMin) {
		this.epitCellMin = epitCellMin;
	}

	public double getEpitCellMax() {
		return epitCellMax;
	}

	public void setEpitCellMax(double epitCellMax) {
		this.epitCellMax = epitCellMax;
	}

	public double getBareNucleiMin() {
		return bareNucleiMin;
	}

	public void setBareNucleiMin(double bareNucleiMin) {
		this.bareNucleiMin = bareNucleiMin;
	}

	public double getBareNucleiMax() {
		return bareNucleiMax;
	}

	public void setBareNucleiMax(double bareNucleiMax) {
		this.bareNucleiMax = bareNucleiMax;
	}

	public double getBlandChromoMin() {
		return blandChromoMin;
	}

	public void setBlandChromoMin(double blandChromoMin) {
		this.blandChromoMin = blandChromoMin;
	}

	public double getBlandChromoMax() {
		return blandChromoMax;
	}

	public void setBlandChromoMax(double blandChromoMax) {
		this.blandChromoMax = blandChromoMax;
	}

	public double getnNucleoliMin() {
		return nNucleoliMin;
	}

	public void setnNucleoliMin(double nNucleoliMin) {
		this.nNucleoliMin = nNucleoliMin;
	}

	public double getnNucleoliMax() {
		return nNucleoliMax;
	}

	public void setnNucleoliMax(double nNucleoliMax) {
		this.nNucleoliMax = nNucleoliMax;
	}

	public double getMitosesMax() {
		return mitosesMax;
	}

	public void setMitosesMax(double mitosesMax) {
		this.mitosesMax = mitosesMax;
	}

	public double getMitosesMin() {
		return mitosesMin;
	}

	public void setMitosesMin(double mitosesMin) {
		this.mitosesMin = mitosesMin;
	}

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append("Summary [clumpThicknessMax=").append(clumpThicknessMax).append(", clumpThicknessMin=")
				.append(clumpThicknessMin).append(", cellSizeMin=").append(cellSizeMin).append(", cellSizeMax=")
				.append(cellSizeMax).append(", cellShapesMax=").append(cellShapesMax).append(", cellShapeMin=")
				.append(cellShapeMin).append(", margAdeisonMin=").append(margAdeisonMin).append(", margAdeisonMax=")
				.append(margAdeisonMax).append(", epitCellMin=").append(epitCellMin).append(", epitCellMax=")
				.append(epitCellMax).append(", bareNucleiMin=").append(bareNucleiMin).append(", bareNucleiMax=")
				.append(bareNucleiMax).append(", blandChromoMin=").append(blandChromoMin).append(", blandChromoMax=")
				.append(blandChromoMax).append(", nNucleoliMin=").append(nNucleoliMin).append(", nNucleoliMax=")
				.append(nNucleoliMax).append(", mitosesMax=").append(mitosesMax).append(", mitosesMin=")
				.append(mitosesMin).append("]");
		return builder.toString();
	}

}

class PatientData {

	private String patientId;
	private double clumpThickness;
	private double cellSize;
	private double cellShape;
	private double margAdeison;
	private double epitCellSize;
	private double bareNuclei;
	private double blandChromo;
	private double nNucleoli;
	private double mitoses;
	private TumorClass tumorClass;

	public String getPatientId() {
		return patientId;
	}

	public void setPatientId(String patientId) {
		this.patientId = patientId;
	}

	public TumorClass getTumorClass() {
		return tumorClass;
	}

	public void setTumorClass(TumorClass tumorClass) {
		this.tumorClass = tumorClass;
	}

	public double getClumpThickness() {
		return clumpThickness;
	}

	public void setClumpThickness(double clumpThickness) {
		this.clumpThickness = clumpThickness;
	}

	public double getCellSize() {
		return cellSize;
	}

	public void setCellSize(double cellSize) {
		this.cellSize = cellSize;
	}

	public double getCellShape() {
		return cellShape;
	}

	public void setCellShape(double cellShape) {
		this.cellShape = cellShape;
	}

	public double getMargAdeison() {
		return margAdeison;
	}

	public void setMargAdeison(double margAdeison) {
		this.margAdeison = margAdeison;
	}

	public double getEpitCellSize() {
		return epitCellSize;
	}

	public void setEpitCellSize(double epitCellSize) {
		this.epitCellSize = epitCellSize;
	}

	public double getBareNuclei() {
		return bareNuclei;
	}

	public void setBareNuclei(double bareNuclei) {
		this.bareNuclei = bareNuclei;
	}

	public double getBlandChromo() {
		return blandChromo;
	}

	public void setBlandChromo(double blandChromo) {
		this.blandChromo = blandChromo;
	}

	public double getnNucleoli() {
		return nNucleoli;
	}

	public void setnNucleoli(double nNucleoli) {
		this.nNucleoli = nNucleoli;
	}

	public double getMitoses() {
		return mitoses;
	}

	public void setMitoses(double mitoses) {
		this.mitoses = mitoses;
	}

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append("PatientData [patientId=").append(patientId).append(", clumpThickness=").append(clumpThickness)
				.append(", cellSize=").append(cellSize).append(", cellShape=").append(cellShape)
				.append(", margAdeison=").append(margAdeison).append(", epitCellSize=").append(epitCellSize)
				.append(", bareNuclei=").append(bareNuclei).append(", blandChromo=").append(blandChromo)
				.append(", nNucleoli=").append(nNucleoli).append(", mitoses=").append(mitoses).append(", tumorClass=")
				.append(tumorClass).append("]");
		return builder.toString();
	}

}

enum TumorClass {
	BENIGN(0), MALIGN(1);
	private int number;

	private TumorClass(int num) {
		this.number = num;
	}

	public int getClassNumber() {
		return this.number;
	}

	public static TumorClass fromName(final String name) {

		if (name.contains("malign")) {
			return TumorClass.MALIGN;
		} else if (name.contains("benign")) {
			return TumorClass.BENIGN;
		} else {
			return null;
		}
	}

	public TumorClass fromNumber(int number) {
		return number == 0 ? TumorClass.BENIGN : TumorClass.MALIGN;
	}

}
