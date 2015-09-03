package algorithms;

import gadgets.DataContainer;
import gadgets.ThresholdMode;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.TwinCursor;
import net.imglib2.type.logic.BitType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.Views;
import results.ResultHandler;

import java.util.List;
import java.util.ArrayList;

import ij.IJ;

/**
 * This algorithm calculates the Gamma-norm Image Colocalization Analysis (GICA).
 * TODO: reference the publication, some short introduction here?
 *
 * @param <T>
 */
public class GammaNormColocalization<T extends RealType< T >> extends Algorithm<T> {


    // this should be user-settable, but where?
    final static int defaultNStat = 50;
    final static int defautlSampleFactor = 5;
    final static int defaultBootstrap = 25;


    // caching for fast random numbers
    // TODO: this works o.k., but could be done better
    static private final double [] rndCache;
    static private final int rndMax = 2048*2048;
    static {
	rndCache = new double [rndMax];
	for (int i=0; i<rndMax; i++)
	    rndCache[i] =  Math.random();
    }


    // Measurement values
    private class Meas {
	// for each ROI: average, variance, calc. threshold, pixels over that threshold
	double avr1=0, var1=0, thr1=0, rat1=0, pxl1=0;
	double avr2=0, var2=0, thr2=0, rat2=0, pxl2=0;
	// measurements
	double gammaAvr, gammaErr;
    
    }

    private Meas meas = new Meas();

    public GammaNormColocalization() {
	    super("GICA coloc");
    }

    @Override
    public void execute(DataContainer<T> container)
	    throws MissingPreconditionException {
	
	// get the two images for the calculation
	RandomAccessible<T> img1 = container.getSourceImage1();
	RandomAccessible<T> img2 = container.getSourceImage2();
	RandomAccessibleInterval<BitType> mask = container.getMask();

	TwinCursor<T> cursor = new TwinCursor<T>(img1.randomAccess(),
	    img2.randomAccess(), Views.iterable(mask).localizingCursor());


	// compute threshold
	final double fac = 3;	// TODO: This needs to be user-settable

	meas.avr1 = container.getMeanCh1();
	meas.avr2 = container.getMeanCh2();
	meas.var1 = container.getVarianceCh1();
	meas.var2 = container.getVarianceCh2();
	meas.thr1 = meas.avr1 + fac * Math.sqrt( meas.var1 );
	meas.thr2 = meas.avr2 + fac * Math.sqrt( meas.var2 );

	// compute which pixels are over threshold
	List<Byte> gammaCacheCh1 = new ArrayList<Byte>(); 
	List<Byte> gammaCacheCh2 = new ArrayList<Byte>(); 

	int ot1=0,ot2=0,count=0;
	while ( cursor.hasNext() ) {
	    cursor.fwd();
	    
	    T type1 = cursor.getFirst();
	    T type2 = cursor.getSecond();
	    
	    double ch1 = type1.getRealDouble(); // o.k. this works because DataContainer forces 
	    double ch2 = type2.getRealDouble(); // T to be RealType, not only Type...

	    gammaCacheCh1.add( (byte)((ch1 >= meas.thr1) ? (1) : (0)));
	    gammaCacheCh2.add( (byte)((ch2 >= meas.thr2) ? (1) : (0)));

	    if (ch1 >= meas.thr1) ot1++;
	    if (ch2 >= meas.thr2) ot2++;
	    count++;

	}
	
	meas.rat1 = ((double)ot1)/count; meas.pxl1 = count;
	meas.rat2 = ((double)ot2)/count; meas.pxl2 = count;

	// perform the gamma calculation
	byte [] gammaCh1 = unbox( gammaCacheCh1 );
	byte [] gammaCh2 = unbox( gammaCacheCh2 );

	IJ.log("Starting GICA computation");

	double [] gm = calculateGammaMeasurement( gammaCh1, gammaCh2, 
	    defautlSampleFactor, defaultNStat, defaultBootstrap); 

	// store measurement
	meas.gammaAvr = gm[0];
	meas.gammaErr = gm[1];

    }

    @Override
    public void processResults(ResultHandler<T> handler) {
        super.processResults(handler);
        handler.handleValue( "GICA gamma value", meas.gammaAvr, 3 );
        handler.handleValue( "GICA gamma error", meas.gammaErr, 3 );
    }


    // ------ Implementation of the algorithm ------

    /** calculates the gamma measurement */
    double [] calculateGammaMeasurement( byte [] gCh1, byte [] gCh2, 
	int sampleFactor, int nStat, final int bootstrap ) {

	// get sum and col
	byte [] gSum = sumGamma( gCh1, gCh2 );
	byte [] gCol = colGamma( gCh1, gCh2 );

	double [] val= new double[bootstrap];

	for (int i=0; i<bootstrap; i++) {
	    IJ.showProgress(i,bootstrap);
	    
	    // ch1 <-> ch2
	    val[i] += Math.pow( calculateGammaCrossCorr( gCh1, gCh2, sampleFactor, nStat ), 2);

	    // each channel with sum
	    val[i] += Math.pow( calculateGammaCrossCorr( gCh1, gSum, sampleFactor, nStat ), 2);
	    val[i] += Math.pow( calculateGammaCrossCorr( gCh2, gSum, sampleFactor, nStat ), 2);

	    // col with sum
	    val[i] += Math.pow( calculateGammaCrossCorr( gCol, gSum, sampleFactor, nStat ), 2);
	    
	    // calc vector len
	    val[i] = Math.sqrt( val[i] );
	}
	
	// ... gamma norms average
	double resAvr=0;
	for (double i : val) 
	    resAvr+=i/val.length;
	
	// .... gamma norms variance
	double resVar=0;
	for (double i : val) 
	    resVar+=Math.pow( i-resAvr ,2 )/ (val.length-1);
	
	return new double [] { resAvr, Math.sqrt( resVar ) };

    }



    /** calculates the cross-correlation portion of two gamma norms */
    double calculateGammaCrossCorr( byte [] valI, byte [] valJ, 
	final int sampleFactor, final int nStat ) {
	
	if (valI.length!=valJ.length)
	    throw new RuntimeException("Inputs have to have the same length");
	
	final int nSample = sampleFactor * valI.length;
	    
	// create #nStat random subsets
	float [] sumI = new float[nStat];
	float [] sumJ = new float[nStat];

	int offS = (int)(Math.random()*rndMax);
	for (int i=0; i<nStat;i++)
	for (int j=0; j<nSample;j++) {
	    int pos = (int)(rndCache[(i*nStat+j*3+offS)%rndMax]*valI.length);
	    sumI[i] += valI[pos];
	    sumJ[i] += valJ[pos];
	} 
	   
	// calculate the average
	float avrI=0, avrJ=0;
	for (float i : sumI) avrI+=(i/nStat);
	for (float j : sumJ) avrJ+=(j/nStat);

	// calculate the cross- and auto-correlation
	float varIJ=0, varI=0, varJ=0;
	for (int i=0; i<sumI.length; i++) {
	    varIJ+=(sumI[i]-avrI)*(sumJ[i]-avrJ);
	    varI+=Math.pow(sumI[i]-avrI,2);
	    varJ+=Math.pow(sumJ[i]-avrJ,2);
	}

	    
	// return the result
	double res =0;
	if (( Math.abs(varJ)>0.001 )&&(Math.abs(varI)>0.001))
	    res = varIJ / (float)(Math.sqrt(varI) * Math.sqrt(varJ));
	
	return res;
    }

    // ------ Internal helper functions ------

    /** Unbox a list of bytes to a primitive byte array */
    static byte [] unbox( List<Byte> in ) {
	byte [] ret = new byte[ in.size() ];
	for (int i=0; i<ret.length; i++)
	    ret[i] = in.get(i);
	return ret;
    }

    /** Point-wise sum of gammas from multiple channels */
    static byte [] sumGamma( byte [] ... in ) {
	
	if ( in == null || in.length==0 ) 
	    throw new NullPointerException("Cannot sum empty elements");
	
	byte [] ret = new byte[ in[0].length ];
	
	// loop all norms
	for ( byte [] cur : in ) {
	    
	    if ( cur.length != ret.length )
		throw new IndexOutOfBoundsException("All input has to have the same length");

	    // sum up each norm
	    for (int i=0; i<ret.length; i++)
		ret[i]+=cur[i];

	}

	return ret;
    }


    /** Point-wise coloc gamma computation */
    static byte [] colGamma( byte [] ... in ) {
    
	if (in == null || in.length==0 ) 
	    throw new NullPointerException("Cannot col empty elements");
	
	byte [] ret = new byte[ in[0].length ];

	// set all gammas to 1
	for ( int i=0; i<ret.length; i++)
	    ret[i] = 1;

	// loop all input gammas
	for ( byte [] cur : in ) {
	    
	    if ( cur.length != ret.length )
		throw new IndexOutOfBoundsException("All input has to have the same length");
	    
	    // set gammas to 0 if they are 0 in any input data
	    for (int i=0; i<ret.length; i++)
		if ( cur[i] == 0 ) ret[i]=0;

	}
	     
	return ret;
    }




}
