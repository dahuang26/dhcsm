from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from scipy import interpolate
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import pickle as pk

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('demo.html')

@app.route('/submit', methods=['POST'])
def submit():
    #SN = request.form['story number']
    #SW = request.form['story weight']
    Design=np.array([5, 5.0043, 161.3736, 4.5070, 2088.3])
    N_st=6
    h1=240
    h2=144
    E_clt=1800
    E_pt=29000
    M=7.4
    fy_pt=105
    g=386
    dratio=0.04
    DOF=N_st+1
    print(N_st)

    def getmodel_GA(N_st=None, h1=None, h2=None, E_clt=None, E_pt=None,m=None,fy_pt=None,Design=None):
        panel = {'t':None,'b':None,'h':None,'E':None,'G':None,'I':None,'H':None,'V':None}
        PT={'a':None,'F_ini':None,'E':None,'l':None,'k_e':None,'k_y':None,'d_y':None,'d':None,'v':None,'fy':None,'Fy':None}
        UFP={'k0':None,'k_ini':None,'k_y':None,'d_y1':None}
        model={"panel":panel,'PT':PT,'UFP':UFP}
        H = h1 + (N_st - 1) * h2#total building height
        #get model porperties
        model['panel']['t']=Design[3] #wall thickness
        model['panel']['h']=H   # wall height
        model['panel']['b']=H/Design[0] # wall width
        model['panel']['E']=E_clt # wall elastic modulus
        model['panel']['G']=1 #wall shear modulus
        model['panel']['I']=model['panel']['t']*model['panel']['b']**3/12 #moment of inertia I=b*L^3/12
        model['panel']['H']=H
        model['panel']['V']=model['panel']['b']*model['panel']['t']*model['panel']['H']
        model['PT']['a']=Design[1] #area of PT bar
        model['PT']['F_ini']=Design[2] #initial force of PT
        model['PT']['E']=E_pt  #E pt
        model['PT']['l']=H #PT length
        model['PT']['k_e']=model['PT']['E']*model['PT']['a']/model['PT']['l']
        model['PT']['k_y']=0.00904*model['PT']['E']*model['PT']['a']/model['PT']['l']
        model['PT']['d_y']=0.02 #yielded displacement
        model['PT']['d']=model['panel']['b']/2#displacement to the rotational pivot point
        model['PT']['v']=model['PT']['a']*H #PTnumber
        model['PT']['fy']=fy_pt
        model['PT']['Fy']=model['PT']['fy']*model['PT']['a']
        model['UFP']['k0']=4*model['panel']['E']*model['panel']['I']/h1   #number of UFP
        model['UFP']['k_ini']=Design[4]  #initial stiffness 
        model['UFP']['k_y']= model['UFP']['k_ini']/10   #yielded stiffness
        model['UFP']['d_y1']=fy_pt/Design[4]/model['panel']['b']   #flag shape hysteresis's first yielded displacement
        model['panel']['dy0']=model['PT']['F_ini']*model['panel']['b']/3/ model['UFP']['k0']
        return model
 
    def getMK_Nst(model=None, N_st=None,M=None,h1=None,h2=None):
        DOF =N_st + 1
        h=np.zeros(N_st)
        for i in range (0,N_st):
            h[i]=h1+h2*i
        #get global m(mass) and k(stiffness)
        m =np.zeros([DOF, DOF])
        k =np.zeros([DOF, DOF])
        for i in range (0, N_st):
            m[i,i]=M
            m[i,N_st]=M*h[i]
            m[N_st,i]=M*h[i]
            if i == 0 and N_st != 1:
                k[i,i]=3*model['panel']['E']*model['panel']['I']/h[i]**3+3*model['panel']['E']*model['panel']['I']/(h[i+1]-h[i])**3
                k[i,i+1]=-3*model['panel']['E']*model['panel']['I']/(h[i+1]-h[i])**3
                k[i+1,i]=k[i,i+1]
            if i==N_st-1 and N_st !=1:
                k[i,i]=3*model['panel']['E']*model['panel']['I']/(h[i]-h[i-1])**3
                k[i,i+1]=0
                k[i+1,i]=k[i,i+1]
            if i!=0 and i!=N_st-1:
                k[i,i]=3*model['panel']['E']*model['panel']['I']/(h[i+1]-h[i])**3+3*model['panel']['E']*model['panel']['I']/(h[i]-h[i-1])**3
                k[i,i+1]=-3*model['panel']['E']*model['panel']['I']/(h[i+1]-h[i])**3
                k[i+1,i]=k[i,i+1]
        m_r = np.zeros([DOF, 1])
        for ii in range (0, N_st):
            m_r[ii,0]=M*h[ii]**2
        m[DOF-1, DOF-1]= sum(m_r)
        return m,k

    def currentK_MDOF(K0=None, Ky=None, dy=None, U=None, V=None, X0=None, DOF=None):
        #this is the function to get the current stiffness of MDOF rocking wall system
        #K0 initial stiffness matrix
        #Ky yielded stiffness matrix
        #dy yielded displacement
        #U current displacement vector
        #V current velocity vector
        #X0 initial location
        #DOF numbers of DOFs
        if abs(U[1] - X0) > dy:
            K = Ky.copy()
        else:
            K = K0.copy()
        if K[DOF-1, DOF-1] == Ky[DOF-1, DOF-1] and V[1] * V[0] <= 0:
            K = K0.copy()
            if V[1] <= 0:
                X0 = U[1] - dy
            else:
                X0 = U[1] + dy
        else:
            K = K
        return K, X0

    def NB_GA(Eq=None, DOF=None, m=None, k0=None, k01=None, k_y1=None, dt=None, g=None, dratio=None, model=None,h1=None,h2=None,Target=None,price=None):
        x01=0
        x02=0
        N_st = DOF-1
        h=np.zeros(N_st)
        S=np.zeros(len(Eq))
        for i in range (0,N_st):
            h[i]=h1+h2*i
        #forve vector
        for jj in range (0,10):
            Fv = np.zeros([DOF, len(Eq[jj])])
            F_v = np.zeros([DOF, len(Eq[jj])])
            for i in range (0,DOF-1):
                Fv[i,:]= -m[i, i] *np.squeeze(Eq[jj])*g
                F_v[i,:]= -m[i,i]*np.squeeze(Eq[jj])*g*h[i]
            Fv[DOF-1, :]= sum(F_v)
            n = len(Fv.T)
            #time =np.arange(0,n*dt[jj],dt[jj])
            k = k0
            #newmark beta method
            beta = 1 / 4
            #rayleigh damping
            #simplified damping 
            c = dratio * 2 * (m * k0) ** (1 / 2)
            X =np.zeros([DOF, n])
            V = np.zeros([DOF, n])
            A = np.zeros([DOF, n])
            Fs = np.zeros([DOF, n])
            U=np.zeros([2,1])
            v=np.zeros([2,1])
            Fs[:,0]= 0
            a = m / beta / dt[jj] + c / 2 / beta
            b = m / 2 / beta - c * dt[jj] * (1 - 1 / 4 / beta)
            ki_bar = k0 + m / beta / dt[jj] ** 2 + c / 2 / beta / dt[jj]
            for ii in range (0,n-1):
            #calculate  dFi-bar
                dFi = Fv[:,ii+1] - Fv[:,ii]
                dFi_bar = -dFi + np.dot(a, V[:, ii]) + np.dot(b , A[:,ii])
            #find dx using dFi-bar/ki-bar
                dX=np.linalg.solve(ki_bar,dFi_bar)
            #find dv using formula
                dV = dX / 2 / beta / dt[jj] - V[:,ii]/ 2 / beta + (1 - 1 / 4 / beta) * dt[jj] * A[:,ii]
            #find da
                dA = dX / beta / dt[jj] ** 2 - V[:,ii] / beta / dt[jj] - A[:,ii] / 2 / beta
            #find Xii+1
                X[:,ii+1] = X[:,ii]+ dX
            #find Vii+1
                V[:,ii+1]= V[:,ii] + dV
            #find Aii+1 using equation of motion
                A[:,ii+1]= A[:,ii] + dA
            #A(:,ii+1)=pinv(m)*(Fv(:,ii+1)-c*V(:,ii+1)-k*X(:,ii+1));
            #get current displacement vector
                U[:,0]= X[DOF-1, ii:ii + 2]
                v[:,0]= V[DOF-1, ii:ii + 2]
                dx = U[1, 0] - U[0, 0]
                #get current stiffness
                if abs(U[1,0]) < model['panel']['dy0']:
                    k = k0
                    ki_bar = k0 + m / beta / dt[jj] ** 2 + c / 2 / beta / dt[jj]
                    x01 = model['panel']['dy0'] + (model['UFP']['d_y1'] - model['panel']['dy0']) / 2
                    x02 = -model['panel']['dy0'] - (model['UFP']['d_y1'] - model['panel']['dy0']) / 2
                    Fs[:,ii+1] = Fs[:,ii] + np.dot(k0 , dX)
                if abs(U[1,0]) > model['panel']['dy0'] and np.sign(U[1,0]) == 1:
                    k_temp1, x01 = currentK_MDOF(k01, k_y1, (model['UFP']['d_y1'] - model['panel']['dy0']) / 2, U, v, x01, DOF)
                    ki_bar = k_temp1 + m / beta / dt [jj]** 2 + c / 2 / beta / dt[jj]
                    Fs[:,ii+1] = Fs[:,ii] + np.dot(k_temp1 , dX)
                #         if U(2,1)>model.PT.d_y/model.PT.d
                #             
                #             
                #             [k,x01]=currentK_MDOF(k01,k_y2,(model.UFP.d_y1-model.panel.dy0)/2,U,v,x01,DOF);
                #             
                #             ki_bar=k+m/beta/dt^2+c/2/beta/dt;
                #         end
                elif abs(U[1, 0]) >model['panel']['dy0'] and np.sign(U[1, 0]) == -1:
                    k_temp2, x02 = currentK_MDOF(k01, k_y1, (model['UFP']['d_y1'] - model['panel']['dy0']) / 2, U, v, x02, DOF)
                    ki_bar = k_temp2 + m / beta / dt[jj] ** 2 + c / 2 / beta / dt[jj]
                    Fs[:,ii+1] = Fs[:,ii] + np.dot(k_temp2 , dX)
                #         if U(2,1)<-model.PT.d_y/model.PT.d
                #             
                #             
                #             [k,x02]=currentK_MDOF(k01,k_y2,(model.UFP.d_y1-model.panel.dy0)/2,U,v,x01,DOF);
                #             
                #             ki_bar=k+m/beta/dt^2+c/2/beta/dt;
                #         end
                if abs(U[1, 0]) <= model['panel']['dy0']: 
                #Fs(jj,ii)=k_ro(jj,1)*U(jj,1);
                    Fs[DOF-1, ii + 1] = k0[DOF-1, DOF-1] * U[1,0]
                if U[0, 0] * dx >= 0 and abs(U[0,0]) <= model['panel']['dy0'] and abs(U[1, 0]) > model['panel']['dy0'] and np.sign(U[0,0]) == 1:
                    Fs[DOF-1, ii + 1]= k01[DOF-1, DOF-1] * U[1, 0] + k0[DOF-1, DOF-1] * model['panel']['dy0'] - k01[DOF-1, DOF-1] * model['panel']['dy0']
                elif U[0, 0] * dx >= 0 and abs(U[0, 0]) <= model['panel']['dy0'] and abs(U[1, 0]) > model['panel']['dy0'] and np.sign(U[0, 0]) == -1:
                    Fs[DOF-1, ii + 1]= k01[DOF-1, DOF-1] * U[1, 0] - (k0[DOF-1, DOF-1] * model['panel']['dy0'] - k01[DOF-1, DOF-1] * model['panel']['dy0'])
                elif U[0, 0] * dx < 0 and abs(U[0, 0]) > model['panel']['dy0'] and abs(U[1, 0]) < model['panel']['dy0']:
                    Fs[DOF-1, ii + 1]= k0[DOF-1, DOF-1] * U[1, 0]
                A[:,ii+1]= np.dot(np.linalg.inv(m) , (Fv[:,ii+1] - np.dot(c , V[:,ii+1])- Fs[:,ii+1]))
            roof_x = X[DOF-1, :] * model['panel']['H'] + X[DOF - 2, :]    
            Drift_roof=max(abs(roof_x))/model['panel']['H']
            if Drift_roof <= Target [0]:
                S[jj]=1
            else:
                S[jj]=0
            Prob=sum(S)/len(S)
            if Prob< Target[1]:
                Score = np.inf
            elif model['panel']['b']**2*model['PT']['F_ini']/(12*k[DOF-1,DOF-1])-1 >0:
                Score=np.inf
            elif model['UFP']['k_ini']*model['UFP']['d_y1']*model['panel']['b']-0.5*model['PT']['Fy']>0:
                Score=np.inf
            elif model['PT']['F_ini']/model['PT']['Fy']>0.5:
                Score=np.inf
            else:
                Score=1*(price[0]*model['panel']['V']+price[1]*model['PT']['v'])     
        return Score

    def f(Design):
        model_design=getmodel_GA(N_st,h1,h2,E_clt,E_pt,M,fy_pt,Design)
        x_c=0; #estimated compression distance .if assume pt is always at center of wall then x_c=0
        #k_ro is the initial rotational stiffness before max stress
        k_ro=model_design['UFP']['k0']
        #k_ro1 initial rotational stiffness
        k_ro1=((model_design['PT']['k_e']*model_design['PT']['d']-x_c)**2+\
            (2/3*model_design['PT']['k_e']*(model_design['PT']['d']-x_c)*x_c)+\
                    model_design['UFP']['k_ini']*model_design['panel']['b']**2)/2
    
        # k_ro2 rotational stiffness after UFP yielded
        k_ro2=((model_design['PT']['k_e']*model_design['PT']['d']-x_c)**2+\
            (2/3*model_design['PT']['k_e']*(model_design['PT']['d']-x_c)*x_c)+\
                    model_design['UFP']['k_y']*model_design['panel']['b']**2)/2  
            
        m,k =getMK_Nst(model_design, N_st,M,h1,h2)
        
        k0=k.copy()
        k0[DOF-1,DOF-1]=k_ro
        
        k01=k.copy()
        k01[DOF-1,DOF-1]=k_ro1
        
        k_y1=k.copy()
        k_y1[DOF-1,DOF-1]=k_ro2
        #Score=np.array([0])
        
        Socre_SLE=NB_GA(EQ_SLE, DOF, m, k0, k01, k_y1, dt, g, dratio, model_design,h1,h2,Target_SLE,price)
        Socre_DBE=NB_GA(EQ_DBE, DOF, m, k0, k01, k_y1, dt, g, dratio, model_design,h1,h2,Target_DBE,price)
        Socre_MCE=NB_GA(EQ_MCE, DOF, m, k0, k01, k_y1, dt, g, dratio, model_design,h1,h2,Target_MCE,price)
        Score=Socre_SLE+Socre_DBE+Socre_MCE
        return Score

    return render_template('success.html', message='You have already submitted the request')
    

if __name__ == '__main__':
    app.debug = True
    app.run()