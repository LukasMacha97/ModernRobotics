import numpy as np
import modern_robotics as mr

def v_log_R(R):
    tr = (np.trace(R) - 1) / 2
    tr = min(max(tr, -1), 1)
    fai = np.arccos(tr)
    if fai == 0:
        w = np.array([0, 0, 0])
    else:
        w = fai / (2 * np.sin(fai)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w


def registration(X, Y, Rn=None, tn=None, Ln=None):
    if X.shape[0] != 3 or Y.shape[0] != 3:
        raise ValueError('Each argument must have exactly three rows.')
    elif X.shape[1] != Y.shape[1]:
        raise ValueError('X and Y must have the same number of columns.')
    elif X.shape[1] < 3 or Y.shape[1] < 3:
        raise ValueError('X and Y must each have 3 or more columns.')
    Npoints = X.shape[1]
    Xbar = np.mean(X, axis=1)
    Ybar = np.mean(Y, axis=1)
    Xtilde = X - np.tile(Xbar, (Npoints, 1)).T
    Ytilde = Y - np.tile(Ybar, (Npoints, 1)).T
    
    H = np.dot(Xtilde, Ytilde.T)
    
    U, S, V = np.linalg.svd(H)

    idx = np.argsort(S)[::-1]
    S = S[idx]
    U = U[:,idx]
    V = V[idx,:].T


    R = np.dot(np.dot(V, np.diag([1, 1, np.linalg.det(np.dot(V, U.T))])), U.T)
    t = Ybar - np.dot(R, Xbar)
    if Rn is not None and tn is not None and Ln is not None:
        ep = np.linalg.norm(t - tn)
        eo = np.linalg.norm(v_log_R(R/Rn))
        et = ep + eo * Ln
        return R, t, ep, eo, et
    else:
        return R, t


def rotation_w(g, theta):

    if theta==0:
        w=np.array([[0],[0],[0]])
        
    else:
        w=1/(2*np.sin(theta)) * np.array([
                                        [g[2,1]-g[1,2]],
                                        [g[0,2]-g[2,0]],
                                        [g[1,0]-g[0,1]] ])

    return w
    

def rotation_theta( g ):
    tr=(np.trace(g[0:3, 0:3]) - 1) / 2 
    if tr>=1:
        tr=1

    elif tr<=-1:

        tr=-1

    theta=np.arccos(tr)

    return theta


def v_log(g):

    fai=rotation_theta(g)
    w=fai*rotation_w(g,fai)
    w = w.reshape((3,))

    if fai == 0:

        p=g[0:3,3]

    else:

        A = np.eye(3)-0.5*mr.VecToso3(w)
        b = (2*np.sin(fai)-fai*(1+np.cos(fai))) / (2*fai*fai*np.sin(fai))

        B = np.dot(np.dot(b,mr.VecToso3(w)),mr.VecToso3(w)) 

        p = np.dot((A+B),g[0:3,3])
    
    kesi=np.r_[w,p]

    return kesi


def a_matrix( kesi, q ):

    w=kesi[0:3]
    v=kesi[3:6]
    bW=np.zeros((6,6))  # change when using ndof
    bW[0:3,0:3]=mr.VecToso3(w)
    bW[3:6,0:3]=mr.VecToso3(v)
    bW[3:6,3:6]=mr.VecToso3(w)
    n=np.linalg.norm(w)
    t=n*q
    if n==0:
        aM=q*np.eye(6)
    else:
        a0 = q * np.eye(6)  # change when using different ndof

        a10 = ( ( 4 - t * np.sin(t) - 4 * np.cos(t) ) / 2 / n**2 )
        a11 = np.dot(a10,bW) 

        a20 = ( ( 4 * t - 5 * np.sin(t) + t * np.cos(t) ) / 2 / n**3 )
        a21 = np.dot(np.dot(a20,bW),bW)

        a30 = ((2-t*np.sin(t)-2*np.cos(t))/2/n**4)
        a31 = np.dot(np.dot(np.dot(a30,bW),bW),bW)

        a40 = ((2*t-3*np.sin(t)+t*np.cos(t))/2/n**5)
        a41 = np.dot(np.dot(np.dot(np.dot(a40,bW),bW),bW),bW)

        aM = a0 + a11 + a21 + a31 + a41

    return aM

def a_matrix_st( kesi ):

    w=kesi[0:3]
    v=kesi[3:6]
    bW=np.zeros((6,6))
    bW[0:3,0:3]=mr.VecToso3(w)
    bW[3:6,0:3]=mr.VecToso3(v)
    bW[3:6,3:6]=mr.VecToso3(w)
    t=np.linalg.norm(w)

    if t==0:

        aM = np.eye(6)

    else:

        a0 = np.eye(6)

        a10 = ( ( 4 - t * np.sin(t) - 4 * np.cos(t) ) / 2 / t**2 )
        a11 = np.dot(a10,bW) 

        a20 = ( ( 4 * t - 5 * np.sin(t) + t * np.cos(t) ) / 2 / t**3 )
        a21 = np.dot(np.dot(a20,bW),bW)

        a30 = ((2-t*np.sin(t)-2*np.cos(t))/2/t**4)
        a31 = np.dot(np.dot(np.dot(a30,bW),bW),bW)

        a40 = ((2*t-3*np.sin(t)+t*np.cos(t))/2/t**5)
        a41 = np.dot(np.dot(np.dot(np.dot(a40,bW),bW),bW),bW)

        aM = a0 + a11 + a21 + a31 + a41

    return aM


def d_exp( kesi, theta ):

    J = np.c_[a_matrix(kesi,theta), kesi]

    return J


def se_3_translation( v,theta ):

    T=np.r_[np.c_[np.eye(3),v*theta],np.array([0,0,0,1]).reshape((1,4))]

    return T


def rotation_matrix( w,theta ):

    R= np.eye(3) + mr.VecToso3(w) * np.sin(theta) + np.dot(mr.VecToso3(w), mr.VecToso3(w)) * (1-np.cos(theta))

    return R


def se_3_rotation( w,v,theta ):

    R=rotation_matrix(w,theta)
    p = np.dot(theta*np.eye(3)+(1-np.cos(theta))*mr.VecToso3(w)+(theta-np.sin(theta))* np.dot(mr.VecToso3(w),mr.VecToso3(w)),v)
    T=np.r_[np.c_[R,p.reshape((3,1))], np.array([0,0,0,1]).reshape(1,4)]

    return T
        

def se_3_exp( kesi ):

    n1=np.linalg.norm(kesi[0:3])
    n2=np.linalg.norm(kesi[3:6])

    if n1==0 and n2==0:
        T=np.eye(4)

    elif n1==0:
        T=se_3_translation(kesi[3:6]/n2,n2)

    else:
        T=se_3_rotation(kesi[0:3]/n1,kesi[3:6]/n1,n1)

    return T



def traditional_calibration_6dof(xi0, xist0, vtheta, gm, M):
   
    xi = np.copy(xi0)

    xist = np.copy(xist0)

    M_home = np.eye(4)

    N = np.size(vtheta,0)

    gn = np.zeros((4,4,N))
    dg = np.zeros((4,4,N))
    v_Log = np.zeros((6,N))

    for m in range(0,M):

        for i in range(0,N):


            gn[:,:,i]= mr.FKinSpace(se_3_exp(xist),xi,vtheta[i,:])
            dg[:,:,i] = np.linalg.solve(gn[:,:,i].T, gm[:,:,i].T).T
            v_Log[:,i]= v_log(dg[:,:,i])

        simY = np.zeros((600, 1))

        for i in range(1, N+1):

            simY[6*i - 6: 6*i, 0] = v_Log[:,i-1]


        simJ = np.zeros((600,42))



        for k in range(0,N):

            mat1 = se_3_exp( np.dot(vtheta[k,0], xi[:,0]))
            a_mat1 = a_matrix(xi[:,0], vtheta[k,0])

            mat2 = se_3_exp( np.dot(vtheta[k,1], xi[:,1]))
            a_mat2 = a_matrix(xi[:,1], vtheta[k,1])

            mat3 = se_3_exp( np.dot(vtheta[k,2], xi[:,2]))
            a_mat3 = a_matrix(xi[:,2], vtheta[k,2])

            mat4 = se_3_exp( np.dot(vtheta[k,3], xi[:,3]))
            a_mat4 = a_matrix(xi[:,3], vtheta[k,3])

            mat5 = se_3_exp( np.dot(vtheta[k,4], xi[:,4]))
            a_mat5 = a_matrix(xi[:,4], vtheta[k,4])

            mat6 = se_3_exp( np.dot(vtheta[k,5], xi[:,5]))
            a_mat6 = a_matrix(xi[:,5], vtheta[k,5])

            a_mat_st = a_matrix_st(xist)

            simJ[6*k:6+6*k,0:6  ] = a_mat1
            simJ[6*k:6+6*k,6:12 ] = np.dot(mr.Adjoint(mat1), a_mat2)
            simJ[6*k:6+6*k,12:18] = np.dot(mr.Adjoint(np.linalg.multi_dot([mat1, mat2])), a_mat3)
            simJ[6*k:6+6*k,18:24] = np.dot(mr.Adjoint(np.linalg.multi_dot([mat1, mat2, mat3])), a_mat4)
            simJ[6*k:6+6*k,24:30] = np.dot(mr.Adjoint(np.linalg.multi_dot([mat1, mat2, mat3, mat4])), a_mat5)
            simJ[6*k:6+6*k,30:36] = np.dot(mr.Adjoint(np.linalg.multi_dot([mat1, mat2, mat3, mat4, mat5])), a_mat6)
            simJ[6*k:6+6*k,36:42] = np.dot(mr.Adjoint(np.linalg.multi_dot([mat1, mat2, mat3, mat4, mat5, mat6])), a_mat_st)

        dp = np.dot(np.linalg.pinv(simJ),simY)

        dp = dp.reshape((len(dp),))

        xi[:,0] = (xi[:,0].reshape((6,1))+dp[0:6].reshape((6,1))).reshape((6,))
        xi[0:3,0] = xi[0:3,0] / np.linalg.norm(xi[0:3,0])
        xi[3:6,0] = xi[3:6,0] + np.dot( np.dot(xi[0:3,1].T, xi[3:6,0]) /  np.dot(xi[0:3,0].T, xi[0:3,0] ), xi[0:3,0])

        xi[:,1] = (xi[:,1].reshape((6,1)) + dp[6:12].reshape((6,1))).reshape((6,))
        xi[0:3,1] = xi[0:3,1]/np.linalg.norm(xi[0:3,1])
        xi[3:6,1] = xi[3:6,1] - np.dot( np.dot(xi[0:3,1].T, xi[3:6,1]) / np.dot(xi[0:3,1].T,xi[0:3,1]), xi[0:3,1])

        xi[:,2] = (xi[:,2].reshape((6,1)) + dp[12:18].reshape((6,1))).reshape((6,))
        xi[0:3,2] = xi[0:3,2]/np.linalg.norm(xi[0:3,2])
        xi[3:6,2] = xi[3:6,2] - np.dot( np.dot(xi[0:3,2].T, xi[3:6,2]) / np.dot(xi[0:3,2].T,xi[0:3,2]), xi[0:3,2])

        xi[:,3]= (xi[:,3].reshape((6,1)) + dp[18:24].reshape((6,1))).reshape((6,))
        xi[0:3,3]=xi[0:3,3] / np.linalg.norm(xi[0:3,3])
        xi[3:6,3]= xi[3:6,3] - np.dot(np.dot(xi[0:3,3].T, xi[3:6,3]) / np.dot(xi[0:3,3].T,xi[0:3,3]),xi[0:3,3])

        xi[:,4]= (xi[:,4].reshape((6,1)) + dp[24:30].reshape((6,1))).reshape((6,))
        xi[0:3,4]=xi[0:3,4] / np.linalg.norm(xi[0:3,4])
        xi[3:6,4]= xi[3:6,4] - np.dot(np.dot(xi[0:3,4].T, xi[3:6,4]) / np.dot(xi[0:3,4].T,xi[0:3,4]),xi[0:3,4])

        xi[:,5]= (xi[:,5].reshape((6,1)) + dp[30:36].reshape((6,1))).reshape((6,))
        xi[0:3,5]=xi[0:3,5] / np.linalg.norm(xi[0:3,5])
        xi[3:6,5]= xi[3:6,5] - np.dot(np.dot(xi[0:3,5].T, xi[3:6,5]) / np.dot(xi[0:3,5].T,xi[0:3,5]),xi[0:3,5])

        xist = xist + dp[36:42]

    return xi, xist


def traditional_calibration_5dof(xi0, xist0, vtheta, gm, M):
   
    xi = np.copy(xi0)

    xist = np.copy(xist0)

    N = np.size(vtheta,0)
    
    n = np.size(vtheta,1)

    gn = np.zeros((4,4,N))
    dg = np.zeros((4,4,N))
    v_Log = np.zeros((6,N))

    for m in range(0,M):

        for i in range(0,N):

            gn[:,:,i]= mr.FKinSpace(se_3_exp(xist),xi,vtheta[i,:])
            dg[:,:,i] = np.linalg.solve(gn[:,:,i].T, gm[:,:,i].T).T
            v_Log[:,i]= v_log(dg[:,:,i])

        simY = np.zeros((600, 1))

        for i in range(1, N+1):

            simY[6*i - 6: 6*i, 0] = v_Log[:,i-1]


        simJ = np.zeros((600,36))

        for k in range(0,N):

            mat1 = se_3_exp( np.dot(vtheta[k,0], xi[:,0]))
            a_mat1 = a_matrix(xi[:,0], vtheta[k,0])

            mat2 = se_3_exp( np.dot(vtheta[k,1], xi[:,1]))
            a_mat2 = a_matrix(xi[:,1], vtheta[k,1])

            mat3 = se_3_exp( np.dot(vtheta[k,2], xi[:,2]))
            a_mat3 = a_matrix(xi[:,2], vtheta[k,2])

            mat4 = se_3_exp( np.dot(vtheta[k,3], xi[:,3]))
            a_mat4 = a_matrix(xi[:,3], vtheta[k,3])

            mat5 = se_3_exp( np.dot(vtheta[k,4], xi[:,4]))
            a_mat5 = a_matrix(xi[:,4], vtheta[k,4])

            a_mat_st = a_matrix_st(xist)

            simJ[6*k : 6+6*k, 0:6  ] = a_mat1
            simJ[6*k : 6+6*k, 6:12 ] = np.dot(mr.Adjoint(mat1), a_mat2)
            simJ[6*k : 6+6*k, 12:18] = np.dot(mr.Adjoint(np.linalg.multi_dot([mat1, mat2])), a_mat3)
            simJ[6*k : 6+6*k, 18:24] = np.dot(mr.Adjoint(np.linalg.multi_dot([mat1, mat2, mat3])), a_mat4)
            simJ[6*k : 6+6*k, 24:30] = np.dot(mr.Adjoint(np.linalg.multi_dot([mat1, mat2, mat3, mat4])), a_mat5)
            simJ[6*k : 6+6*k, 30:36] = np.dot(mr.Adjoint(np.linalg.multi_dot([mat1, mat2, mat3, mat4, mat5])), a_mat_st)

        dp = np.dot(np.linalg.pinv(simJ),simY)

        dp = dp.reshape((len(dp),))

        xi[:,0] = (xi[:,0].reshape((6,1)) + dp[0:6].reshape((6,1))).reshape((6,))
        xi[0:3,0] = xi[0:3,0] / np.linalg.norm(xi[0:3,0])
        xi[3:6,0] = xi[3:6,0] + np.dot( np.dot(xi[0:3,1].T, xi[3:6,0]) /  np.dot(xi[0:3,0].T, xi[0:3,0] ), xi[0:3,0])

        xi[:,1] = (xi[:,1].reshape((6,1)) + dp[6:12].reshape((6,1))).reshape((6,))
        xi[0:3,1] = xi[0:3,1]/np.linalg.norm(xi[0:3,1])
        xi[3:6,1] = xi[3:6,1] - np.dot( np.dot(xi[0:3,1].T, xi[3:6,1]) / np.dot(xi[0:3,1].T,xi[0:3,1]), xi[0:3,1])

        xi[:,2] = (xi[:,2].reshape((6,1)) + dp[12:18].reshape((6,1))).reshape((6,))
        xi[0:3,2] = xi[0:3,2]/np.linalg.norm(xi[0:3,2])
        xi[3:6,2] = xi[3:6,2] - np.dot( np.dot(xi[0:3,2].T, xi[3:6,2]) / np.dot(xi[0:3,2].T,xi[0:3,2]), xi[0:3,2])

        xi[:,3]= (xi[:,3].reshape((6,1)) + dp[18:24].reshape((6,1))).reshape((6,))
        xi[0:3,3]=xi[0:3,3] / np.linalg.norm(xi[0:3,3])
        xi[3:6,3]= xi[3:6,3] - np.dot(np.dot(xi[0:3,3].T, xi[3:6,3]) / np.dot(xi[0:3,3].T,xi[0:3,3]),xi[0:3,3])

        xi[:,4]= (xi[:,4].reshape((6,1)) + dp[24:30].reshape((6,1))).reshape((6,))
        xi[0:3,4]=xi[0:3,4] / np.linalg.norm(xi[0:3,4])
        xi[3:6,4]= xi[3:6,4] - np.dot(np.dot(xi[0:3,4].T, xi[3:6,4]) / np.dot(xi[0:3,4].T,xi[0:3,4]),xi[0:3,4])

        # xi[:,5]= (xi[:,5].reshape((6,1)) + dp[30:36].reshape((6,1))).reshape((6,))
        # xi[0:3,5]=xi[0:3,5] / np.linalg.norm(xi[0:3,5])
        # xi[3:6,5]= xi[3:6,5] - np.dot(np.dot(xi[0:3,5].T, xi[3:6,5]) / np.dot(xi[0:3,5].T,xi[0:3,5]),xi[0:3,5])

        xist = xist + dp[30:36]

    return xi, xist



def traditional_calibration_scara(xi0, vtheta, gm, M):

    xi = np.copy(xi0)

    M_home = np.eye(4)

    dq=np.zeros((4,1))
    N=np.size(vtheta,0)

    gn=np.zeros((4,4,N))
    dg=np.zeros((4,4,N))
    v_Log=np.zeros((6,N))

    for i in range(0,N):

        gn[:,:,i]= mr.FKinBody(M_home, xi, vtheta[i,:])
        dg[:,:,i] = np.linalg.solve(gn[:,:,i].T, gm[:,:,i].T).T
        v_Log[:,i]= v_log(dg[:,:,i])

    error=np.zeros((3,1))

    for i in range(0,N):

        error=error+ np.array([
            [np.linalg.norm(v_Log[:,i])],
            [np.linalg.norm(v_Log[3:6,i])],
            [np.linalg.norm(v_Log[0:3,i])]])


    simJ = np.zeros((60,25))
    meanE = np.zeros((3,11))

    meanE[:,0]=(error/N).reshape((3,))

    convergence=np.zeros((10,2))

    for m in range(0,10):

        for i in range(0,N):

            gn[:,:,i]= mr.FKinBody(M_home, xi, vtheta[i,:])
            dg[:,:,i]= np.linalg.solve(gn[:,:,i].T,gm[:,:,i].T).T 
            v_Log[:,i] = v_log(dg[:,:,i])

        simY=np.zeros((6*N,1))

        for i in range(0,N):

            simY[6*(i+1)-6 : 6 * (i+1), 0 ] = v_Log[:,i]
        
        for k in range(0,N-1):

            simJ[0+6*k:6+6*k,0:7] = d_exp(xi[:,0], vtheta[k,0])

            simJ[0+6*k:6+6*k,7:14]= np.dot( mr.Adjoint( se_3_exp( np.dot(vtheta[k,0], xi[:,0])) ), d_exp( xi[:,1], vtheta[k,1] ))

            tempSIMJ= np.dot(mr.Adjoint( np.dot(se_3_exp( np.dot(vtheta[k,0], xi[:,0])), se_3_exp( np.dot(vtheta[k,1], xi[:,1])))), d_exp( xi[:,2], vtheta[k,2]))

            simJ[6*k:6+6*k,14:18] = tempSIMJ[:,3:7]

            simJ[0+6*k:6+6*k,18:25] = np.dot(mr.Adjoint( np.dot(np.dot(se_3_exp( np.dot(vtheta[k,0], xi[:,0]) ), se_3_exp( np.dot( vtheta[k,1], xi[:,1] ) )), se_3_exp( np.dot( vtheta[k,2], xi[:,2] ) )) ),d_exp(xi[:,3],vtheta[k,3]))

            dp = np.linalg.lstsq(simJ, simY, rcond=None)[0]


        xi[:,0] = (xi[:,0].reshape((6,1))+dp[0:6].reshape((6,1))).reshape((6,))
        xi[0:3,0] = xi[0:3,0] / np.linalg.norm(xi[0:3,0])
        xi[3:6,0] = xi[3:6,0] + np.dot( np.dot(xi[0:3,1].T, xi[3:6,0]) /  np.dot(xi[0:3,0].T, xi[0:3,0] ), xi[0:3,0])

        xi[:,1] = (xi[:,1].reshape((6,1)) + dp[7:13].reshape((6,1))).reshape((6,))
        xi[0:3,1] = xi[0:3,1]/np.linalg.norm(xi[0:3,1])
        xi[3:6,1] = xi[3:6,1] - np.dot( np.dot(xi[0:3,1].T, xi[3:6,1]) / np.dot(xi[0:3,1].T,xi[0:3,1]), xi[0:3,1])

        xi[3:6,2]=(xi[3:6,2].reshape((3,1)) + dp[14:17].reshape((3,1))).reshape((3,))
        xi[3:6,2]=xi[3:6,2] / np.linalg.norm(xi[3:6,2])

        xi[:,3]= (xi[:,3].reshape((6,1)) + dp[18:24].reshape((6,1))).reshape((6,))
        xi[0:3,3]=xi[0:3,3] / np.linalg.norm(xi[0:3,3])
        xi[3:6,3]= xi[3:6,3] - np.dot(np.dot(xi[0:3,3].T, xi[3:6,3]) / np.dot(xi[0:3,3].T,xi[0:3,3]),xi[0:3,3])

        vtheta[:,0] = vtheta[:,0]+dp[6]
        vtheta[:,1] = vtheta[:,1]+dp[13]
        vtheta[:,2] = vtheta[:,2]+dp[17]
        vtheta[:,3] = vtheta[:,3]+dp[24]

        dq = dq + np.array([dp[6], dp[13], dp[17], dp[24]])

        for i in range(0,N):
            gn[:,:,i] = mr.FKinBody(M_home, xi, vtheta[i,:])
            dg[:,:,i] = np.linalg.solve(gn[:,:,i].T,gm[:,:,i].T).T
            v_Log[:,i] = v_log(dg[:,:,i])

        error=np.zeros((3,1))

        for i in range(0,N):
            
            error=error + np.r_[ np.r_[ np.linalg.norm(v_Log[:,i]), np.linalg.norm(v_Log[3:6,i]) ], np.linalg.norm(v_Log[0:3,i]) ].reshape((3,1))

        meanE[:, m] = (error/N).reshape((3,))
        convergence[m,0]=np.linalg.norm(simY)
        convergence[m,1]=np.linalg.norm(dp)

    return xi, dq, meanE, convergence


def main(argv, argc):
    pass

if __name__ == "__main__":
    main()