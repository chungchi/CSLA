#include "mex.h"
#include <math.h>

void do_kmeans(int vq_num,int vq_dim,int Knum,int Itnum,double dat[],double rcen[],double Cout[],double Lout[])   
{
	double very_large=1000000000; //3*256^2
	double dis,temp,InvKNum;
	double error=1.0;
    int Lnum;
    int j,k,i,rid,*tnum;
    double *vq_temp,*vq_sum;
 
    vq_temp=mxCalloc(vq_dim,sizeof(double));
    vq_sum=mxCalloc(vq_dim*Knum,sizeof(double));
    tnum=mxCalloc(Knum,sizeof(int));
    Lnum=0;
    
    // mexPrintf("Itnum %d \n",Itnum);
    
	while (error>0.0001 && Lnum<Itnum)
	{
        Lnum=Lnum+1;
		error=0.0;
		for (j=0;j<Knum;j++)
		{
			tnum[j]=0;
		} 
		for (j=0;j<Knum*vq_dim;j++)
		{
			vq_sum[j]=0;
		} 
		for(j=0;j<vq_num;j++)
		{
			temp=very_large;
            for(i=0;i<vq_dim;i++)
               vq_temp[i]=dat[j*vq_dim+i];
			/*Classification*/
			for(k=0;k<Knum;k++)
			{
                dis=0;
                for(i=0;i<vq_dim;i++)
                    dis=dis+(vq_temp[i]-rcen[k*vq_dim+i])*(vq_temp[i]-rcen[k*vq_dim+i]);
				if (dis<temp)
				{
					temp=dis;
					rid=k;
				}
			}
			tnum[rid]=tnum[rid]+1;
			Lout[j]=rid;
            for(i=0;i<vq_dim;i++)
               vq_sum[rid*vq_dim+i]=vq_sum[rid*vq_dim+i]+vq_temp[i];
		}
		for (j=0;j<Knum;j++)
		{
			if (tnum[j]!=0)
			{
				InvKNum=1.0/tnum[j];
                for(i=0;i<vq_dim;i++)
                {
                   Cout[j*vq_dim+i]=vq_sum[j*vq_dim+i]*InvKNum;
				   error=error+fabs(Cout[j*vq_dim+i]-rcen[j*vq_dim+i]);
                }
			}
		} 
        for(i=0;i<vq_dim*Knum;i++)
            rcen[i]=Cout[i];
//      mexPrintf("error is %f /Iteration num is %d \n",error,Lnum);        
	}
//        mexPrintf("Total error is %f /Iteration num is %d \n",error,Lnum);
//     free(vq_temp);
//     free(vq_sum);
//     free(tnum);

}


void mexFunction(
  int nlhs, mxArray *plhs[],
  int nrhs, const mxArray *prhs[] )
{
  int Knum,Itnum,m,n;
  double *dat,*rcen,*Cout,*Lout;
  
  
  //pnum,nnum,dat,label,w
  /* Check for proper number of arguments */
  if (nrhs != 4)
    mexErrMsgTxt("5 inputs in the form (H,Ig,segmask) is required.");
  else if (nlhs != 2)
    mexErrMsgTxt("1 outputs in the form [Fout] is required.");

  /* Create a pointer to a copy of the vector data. */
  Knum = mxGetScalar(prhs[1]);
  Knum=(int)Knum;
  Itnum= mxGetScalar(prhs[2]);
  Itnum=(int)Itnum;
  dat=mxGetPr(prhs[0]);
  rcen=mxGetPr(prhs[3]);
 
//   mexPrintf("N1,N0,Kis %d %d %d\n",N1,N0,K);
  m = mxGetN(prhs[0]); //column
  n = mxGetM(prhs[0]); //row
  /* Create the output Fout  */
  plhs[0] = mxCreateDoubleMatrix(n,Knum,mxREAL);
  plhs[1] = mxCreateDoubleMatrix(m,1,mxREAL);
//   plhs[2] = mxCreateDoubleMatrix(nrow,mcol,mxREAL);
 
  Cout=mxGetPr(plhs[0]);
  Lout = mxGetPr(plhs[1]);
//   Tout = mxGetPr(plhs[2]);
//    mexPrintf("Total num: %d Dim: %d \n",m,n);

 do_kmeans(m,n,Knum,Itnum,dat,rcen,Cout,Lout);

//  free(Fout);
//  free(Bout);
//  free(Tout);
}


