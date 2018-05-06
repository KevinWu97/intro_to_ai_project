import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;

public class SplitTextFiles {

	private static final int FACE_HEIGHT = 70;
	private static final int DIGIT_HEIGHT = 28;
	
	public static void main(String[] args) {
		try {
			if (args.length != 2) {
				System.out.println("Needs 2 arguments: digit/face and test/train");
				System.exit(0);
			}

			String prefix = args[0];
			String type = args[1];

			String fileName = "";
			int height = 0;
			if (prefix.equalsIgnoreCase("face")) {
				fileName = "facedata" + type;
				height = FACE_HEIGHT;
			} else {
				if (type.equalsIgnoreCase("train")) {
					fileName = "trainingimages";
				} else {
					fileName = "testimages";
				}
				height = DIGIT_HEIGHT;
			}

			String fileLoc = prefix + "data_" + type + "_split\\";			

			//Reading from original file
			FileInputStream fis = new FileInputStream(prefix + "data\\" + fileName);
			BufferedInputStream bis = new BufferedInputStream(fis);
			InputStreamReader isr = new InputStreamReader(bis);
			BufferedReader br = new BufferedReader(isr);
			
			//We read from file until it is blank
			//However, for numbers, we do this in chunks of 26
			String content = "";
			int fileIndex = 0;
			do {
				int readLineIndex = 0;

				File dir = new File(fileLoc);
				dir.mkdir();

				String filename = fileLoc + prefix + fileIndex + ".txt";
				File file = new File(filename);

				if(file.createNewFile()) {
					// System.out.println("File successfully created");
				}else {
					// System.out.println("File already exists");
				}
				FileWriter fw = new FileWriter(file, true);
				BufferedWriter bw = new BufferedWriter(fw);



				while(readLineIndex < height && (content = br.readLine()) != null) {	
					bw.write(content + "\n");
					readLineIndex++;
				}

				bw.close();
				fileIndex++;
			} while (content != null);
			br.close();
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
		
	}

}
