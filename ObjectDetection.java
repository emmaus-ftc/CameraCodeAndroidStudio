/*
 * Copyright (c) 2019 OpenFTC Team
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package org.firstinspires.ftc.teamcode;

import com.qualcomm.robotcore.eventloop.opmode.LinearOpMode;
import com.qualcomm.robotcore.eventloop.opmode.TeleOp;

import org.firstinspires.ftc.robotcore.external.hardware.camera.WebcamName;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core; // Nodig voor inRange
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.MatOfPoint; // Nodig voor contouren
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point3;
import org.opencv.core.Rect; // Nodig voor de rechthoek om het object
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc; // Nodig voor cvtColor, inRange, findContours, drawContours
import org.openftc.easyopencv.OpenCvCamera;
import org.openftc.easyopencv.OpenCvCameraFactory;
import org.openftc.easyopencv.OpenCvCameraRotation;
import org.openftc.easyopencv.OpenCvPipeline;
import org.openftc.easyopencv.OpenCvWebcam;

import java.util.ArrayList;
import java.util.List;


/// /////////////////////////////////////////////////////
/// MEET "MEASUREDPIXELWIDTH" IN VISION PIPELINE VOOR AFSTAND DETECTION!!!!!
/// /////////////////////////////////////////////////////


@TeleOp(name="Webcam Color Detect", group="Robot") // Naam op Driver Station
public class ObjectDetection extends LinearOpMode
{
    OpenCvWebcam webcam;

    @Override
    public void runOpMode()
    {
        // Verkrijg de ID van de camera-monitor view op het Driver Station
        int cameraMonitorViewId = hardwareMap.appContext.getResources().getIdentifier("cameraMonitorViewId", "id", hardwareMap.appContext.getPackageName());

        // Initialiseer de webcam met de naam "Webcam 1" uit de hardware configuratie
        // en koppel deze aan de camera monitor view.
        webcam = OpenCvCameraFactory.getInstance().createWebcam(hardwareMap.get(WebcamName.class, "Webcam 1"), cameraMonitorViewId);

        // Maak een nieuwe instantie van je aangepaste pipeline voor kleurdetectie
        // BELANGRIJKE WIJZIGING: Geef de 'webcam' instantie door aan de pipeline
        // WIJZIGING: Gebruik de nieuwe naam voor de pipeline klasse
        GreenSphereDetectionPipeline colorDetectionPipeline = new GreenSphereDetectionPipeline(webcam);
        webcam.setPipeline(colorDetectionPipeline);

        // Stel een timeout in voor het verkrijgen van cameratoestemming
        webcam.setMillisecondsPermissionTimeout(5000);

        // Open de camera asynchroon
        webcam.openCameraDeviceAsync(new OpenCvCamera.AsyncCameraOpenListener()
        {
            @Override
            public void onOpened()
            {
                // Start de camerastream met een resolutie van 320x240 en rechtopstaande rotatie
                webcam.startStreaming(320, 240, OpenCvCameraRotation.UPRIGHT);
                telemetry.addData("Camera Status", "Stream gestart");
                telemetry.update();
            }

            @Override
            public void onError(int errorCode)
            {
                // Toon een foutmelding als de camera niet geopend kon worden
                telemetry.addData("Camera Fout", "Code: " + errorCode);
                telemetry.update();
            }
        });

        telemetry.addLine("Waiting for start");
        telemetry.update();

        // Wacht tot de gebruiker op START drukt op het Driver Station
        waitForStart();

        while (opModeIsActive())
        {
            // Stuur diverse statistieken van de camera en pipeline naar de telemetry
            telemetry.addData("Frame Count", webcam.getFrameCount());
            telemetry.addData("FPS", String.format("%.2f", webcam.getFps()));
            telemetry.addData("Total frame time ms", webcam.getTotalFrameTimeMs());
            telemetry.addData("Pipeline time ms", webcam.getPipelineTimeMs());
            telemetry.addData("Overhead time ms", webcam.getOverheadTimeMs());
            telemetry.addData("Theoretical max FPS", webcam.getCurrentPipelineMaxFps());

            // voeg telemetry toe voor de afstand en orientatie van de baluwe objecten
            //telemetry.addData("Blok rotatie (graden)", colorDetectionPipeline.getLastAngle());
            telemetry.addData("Afstand tot blok (cm)", String.format("%.2f", colorDetectionPipeline.getLastDistance()));
            //telemetry.addData("Width van het blok in pixels", colorDetectionPipeline.getPixelWidth());

            // Voeg telemetry toe voor het aantal gedetecteerde blauwe objecten
            telemetry.addData("Blauwe Objecten", colorDetectionPipeline.getNumberOfGreenObjects());
            telemetry.update();

            // Stop de stream als de 'A' knop op gamepad 1 wordt ingedrukt
            if(gamepad1.a)
            {
                webcam.stopStreaming();
                // webcam.closeCameraDevice(); // Optioneel: sluit de camera volledig als je deze wilt overdragen aan een ander vision-systeem
            }

            // Pauzeer de loop om CPU-cycli te besparen
            sleep(100);
        }
        // Stop de stream en sluit de camera wanneer de OpMode eindigt
        webcam.stopStreaming();
        webcam.closeCameraDevice();
    }

    /*
     * Dit is de aangepaste beeldverwerkingspipeline voor het detecteren van een blauw blok.
     * Het converteert het beeld naar HSV, creëert een masker voor blauwe kleuren,
     * vindt contouren en tekent rechthoeken om de gedetecteerde objecten.
     * WIJZIGING: De klasse is hernoemd van SamplePipeline naar BlueBlockDetectionPipeline
     * om naamconflicten te voorkomen.
     */
    class GreenSphereDetectionPipeline extends OpenCvPipeline
    {

        boolean viewPortPaused;

        // Mat objecten voor beeldverwerking. Het hergebruiken hiervan voorkomt geheugenlekken en verbetert de efficiëntie.
        Mat hsv = new Mat();
        Mat mask = new Mat();
        Mat hierarchy = new Mat(); // Nodig voor findContours

        // HSV-waarden voor groen.
        // Hue (kleurtoon): 0-179 (OpenCV schaal)
        // Saturation (verzadiging): 0-255
        // Value (helderheid): 0-250

        Scalar lowerGreen = new Scalar(35, 80, 80);
        Scalar upperGreen = new Scalar(85, 255, 255);


        // Variabele om het aantal gedetecteerde objecten bij te houden
        private volatile int numberOfGreenObjects = 0;
        private volatile double lastRadius = 0;
        private volatile double lastDistance = 0;
        //private volatile double lastPixelWidth = 0;

        // BELANGRIJKE WIJZIGING: Voeg een referentie naar de webcam toe
        private OpenCvWebcam externalWebcam;

        // Constructor om de webcam instantie te ontvangen
        public GreenSphereDetectionPipeline(OpenCvWebcam webcam)
        {
            this.externalWebcam = webcam;
        }

        // Getter voor het aantal gedetecteerde objecten, zodat de OpMode deze kan uitlezen
        public int getNumberOfGreenObjects()
        {
            return numberOfGreenObjects;
        }

        public double getLastRadius() {
            return lastRadius;
        }

        public double getLastDistance() {
            return lastDistance;
        }

//        public double getPixelWidth() {
//           return lastPixelWidth;
//        }

        @Override
        public Mat processFrame(Mat input)
        {
            // Converteer het input frame van RGB naar HSV kleurruimte
            Imgproc.cvtColor(input, hsv, Imgproc.COLOR_RGB2HSV);

            // Creëer een binair masker: witte pixels waar de kleur binnen het blauwe bereik valt, zwart elders.
            Core.inRange(hsv, lowerGreen, upperGreen, mask);

            // Zoek contouren (omtrekken van objecten) in het binaire masker
            List<MatOfPoint> contours = new ArrayList<>();
            // Imgproc.RETR_EXTERNAL haalt alleen de buitenste contouren op
            // Imgproc.CHAIN_APPROX_SIMPLE comprimeert horizontale, verticale en diagonale segmenten
            Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            // Reset het aantal gedetecteerde objecten
            numberOfGreenObjects = 0;
            double maxRadius = 0;

            // Loop door de gevonden contouren
            for (MatOfPoint contour : contours)
            {
                // Bereken de oppervlakte van de contour
                double contourArea = Imgproc.contourArea(contour);

                // Filter contouren op basis van oppervlakte om ruis te verwijderen
                // Pas deze waarden aan op basis van de grootte van je blok en de afstand tot de camera
                if (contourArea > 500) // Minimale oppervlakte van de contour (experimenteer hiermee!)
                {

                    Point center = new Point();
                    float[] radius = new float[1];
                    MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
                    Imgproc.minEnclosingCircle(contour2f, center, radius);

                    double pixelRadius = radius[0];
                    double pixelDiameter = 2 * pixelRadius;

                    double knownDistance = 30.0; // Aantal Centimeter waarvan je de measuredPixelDiameter hebt gemeten
                    double realDiameter = 12.70;
                    double measuredPixelDiameter = 0; // VUL LATER IN ALS JE GAAT MEASUREN!!!!
                    double focalLenght = (measuredPixelDiameter * knownDistance) / realDiameter;

                    double distance = (realDiameter * focalLenght) / pixelDiameter;

                    if (pixelRadius > maxRadius) {
                        maxRadius = pixelRadius;
                        lastDistance = distance;
                        lastRadius = pixelRadius;
                    }

                    Imgproc.circle(input, center, (int) pixelRadius, new Scalar(0, 255, 0), 2);
                    Imgproc.circle(input, center, 3, new Scalar(0, 0, 255), -1);

                    numberOfGreenObjects++;
                    contour2f.release();


                }
                // Geef het geheugen van de contour MatOfPoint vrij
                contour.release();
            }

            // Geef het geheugen van de tijdelijke Mat objecten vrij
            hsv.release();
            mask.release();
            hierarchy.release();

            // Retourneer het originele frame met de getekende rechthoeken
            return input;
        }

        @Override
        public void onViewportTapped()
        {

            viewPortPaused = !viewPortPaused;
            if(viewPortPaused) // Controleer de huidige status van de webcam-viewport
            {
                externalWebcam.pauseViewport();
            }
            else
            {
                externalWebcam.resumeViewport(); // Pauzeer de weergave
            }
        }
    }
}
