import numpy as np
from ngsolve import *
from ngsolve.webgui import Draw
from ngsolve.meshes import Make1DMesh
from pymor.bindings.ngsolve import NGSolveMatrixOperator, NGSolveVectorSpace
from pymor.operators.interface import Operator
from pymor.models.basic import InstationaryModel
from pymor.algorithms.timestepping import ImplicitEulerTimeStepper
from pymor.core.defaults import set_defaults



set_defaults({
    'pymor.algorithms.newton.newton.relax': 1.0,  
    'pymor.algorithms.newton.newton.rtol': 1e-6,
    'pymor.algorithms.newton.newton.atol': 1e-8,
    'pymor.algorithms.newton.newton.maxiter': 100
})
# here create the Operator A (we can plug into InstationaryModel)
# create a new Operator that inherits from Operator

class BrusselatorOperator(Operator): # pyMOR object (that will be computed using ngsolve)
    
    def __init__(self, V_h, a_form, residual_func, jacobian_func, v_space, solver_options=None):

        super().__init__() 
        
        self.linear = False
        self.solver_options = solver_options
        self.source = v_space
        self.range = v_space
        
        self.v_space = v_space
        self.V_h = V_h
        
        self.a_form = a_form
        self.residual_func = residual_func
        self.jacobian_func = jacobian_func

        self.parameters_own = {'beta': 1} # this is the parameter

    # def apply(self, U, mu=None):   # compute the residual

    #     gfu_impl = U.vectors[0].real_part.impl   # get the Gridfunction (ngsolve)
    #     vec_ng = gfu_impl.vec    # get the BaseVector (ngsolve)

    #     current_beta = mu['beta']   # update beta
    
    #     res_vec = self.residual_func(vec_ng, current_beta)  # compute residual in ngsolve, return basevector
    
    #     res_gf = GridFunction(self.V_h)  # wrap into new Gridfunction
    #     res_gf.vec.data = res_vec
    
    #     return self.range.make_array([res_gf])      # make VectorArray (pyMOR)

    def apply(self, U, mu=None):
        current_beta = mu['beta']
        result_list = []
        
        for i in range(len(U)):
            gfu_impl = U.vectors[i].real_part.impl
            vec_ng = gfu_impl.vec
            
            res_vec = self.residual_func(vec_ng, current_beta)
            
            res_gf = GridFunction(self.V_h)
            res_gf.vec.data = res_vec
            result_list.append(res_gf)
            
        return self.range.make_array(result_list)
    
    def jacobian(self, U, mu=None):   # compute jacobian in ngsolve
        
        gfu_impl = U.vectors[0].real_part.impl
        vec_ng = gfu_impl.vec
        
        current_beta = mu['beta']

        mat_ng = self.jacobian_func(vec_ng, current_beta)    
        
        return NGSolveMatrixOperator(mat_ng, range=self.range, source=self.source)



def create_fom(nt=64, h=50, time=9, initial_array=None):
    # create the FOM using ngsolve
    # here define mesh, FE-space, operators

    # here we tell ngsolve what to do when pyMOR calls it


    alpha = 1.0
    beta = Parameter(3.25)  # solve for different beta later
    nu = 0.01
    order = 2               # quadratic Lagrange basis functions

    mesh = Make1DMesh(h)   # mesh on (0,1) into h pieces

    # V_h = V_u * V_v 
    V_h = H1(mesh, order=order, dirichlet="right") * H1(mesh, order=order, dirichlet="right")

    u, v = V_h.TrialFunction()    # represent unknowns 
    phi, psi = V_h.TestFunction() # test functions ("for u and v")


    # This holds the values where the nonlinear operator is evaluated
    w_h = GridFunction(V_h)          # w_h element in V_h with zeros
    u_part, v_part = w_h.components  # seperate u and v part in w_h 

    # generate mass Matrix m 
    m = BilinearForm(V_h)           # m: V_h x V_h to R
    m += (u * phi + v * psi) * dx   # u * phi fills the upper-left block (u part)
                                    # v * psi fills the lower-right block (v part)
    m.Assemble()

    # generate stiffness matrix a
    a = BilinearForm(V_h)          # a: V_h x V_h to R
    a += (nu * grad(u) * grad(phi) + nu * grad(v) * grad(psi)) * dx
    a.Assemble()

    # H01-inner product
    H01 = BilinearForm(V_h)
    H01 += (grad(u) * grad(phi) + grad(v) * grad(psi)) * dx
    H01.Assemble()

    # for the non-lin part we use Newtons method F(u + \delta u) \approx F(u) + F'(u)* \delta u
    # for Newton we need the residual and the Jacobian

    # residual
    res = LinearForm(V_h)

    y_sym = alpha + u_part
    z_sym = (beta / alpha) + v_part

    res += (nu * grad(u_part) * grad(phi) + nu * grad(v_part) * grad(psi)) * dx   # diffusion part for both (grad(u_part)=grad(y_sym)
    res += -(y_sym**2 * z_sym - (beta + 1) * y_sym + alpha) * phi * dx            # nonlinear part for y
    res += -(beta * y_sym - y_sym**2 * z_sym) * psi * dx                          # nonlinear part for z

    # jacobian
    jac = BilinearForm(V_h)

    jac += (nu * grad(u) * grad(phi) + nu * grad(v) * grad(psi)) * dx
    jac += -( (2*y_sym*z_sym - (beta+1))*u + (y_sym**2)*v ) * phi * dx
    jac += -( (beta - 2*y_sym*z_sym)*u - (y_sym**2)*v ) * psi * dx

    # define the functions

    def spatial_residual(vec, beta_val):
        beta.Set(beta_val.item() if hasattr(beta_val, 'item') else float(beta_val))
        w_h.vec.data = vec
        res.Assemble()
        
        return res.vec

    def jacobian(vec, beta_val):
        beta.Set(beta_val.item() if hasattr(beta_val, 'item') else float(beta_val))
        w_h.vec.data = vec
        jac.Assemble()
        
        return jac.mat



    # bind the linear parts

    # define the pyMOR VectorSpace from your NGSolve FE space
    v_space = NGSolveVectorSpace(V_h)    # make it a pyMOR VectorSpace

    m_op = NGSolveMatrixOperator(m.mat, range=v_space, source=v_space)   # pyMOR operator that takes VectorArray as input
    a_op = NGSolveMatrixOperator(a.mat, range=v_space, source=v_space)

    H01_op = NGSolveMatrixOperator(H01.mat, range=v_space, source=v_space)


    # ingridients for the Model

    # create the concrete operator we want (that works in PyMOR but uses ng computations)
    a_br = BrusselatorOperator(
        V_h=V_h, 
        a_form=a_op, 
        residual_func=spatial_residual, 
        jacobian_func=jacobian,
        v_space=v_space,
    )


    # take some function as initial cond

    if initial_array is not None:
        u0_vec = v_space.from_numpy(initial_array)
    else:
        gfu0 = GridFunction(V_h)
        gfu0.components[0].Set(0.1 * cos(1.57 * x))
        gfu0.components[1].Set(0.1 * sin(3.14 * x)) 
        u0_vec = v_space.make_array([gfu0])



    # here first create the FOM with the random initial condition from above
    stepper_optionen = {
        'type': 'newton',
        'rtol': 1e-8,
        'atol': 1e-11,
        'maxiter': 100,
        'line_search': False  
    }
    
    stepper = ImplicitEulerTimeStepper(
        nt=nt, 
        solver_options=stepper_optionen
    )
    
    # this sets the newton-tol for every time step
    # nt_fom = 64 * 100  # 6400 Schritte
    # stepper = ImplicitEulerTimeStepper(nt=nt_fom)
    #stepper = ImplicitEulerTimeStepper(nt=nt)   #,solver_options={'rtol': 1e-6, 'atol': 1e-8, 'maxiter': 50  }) # take 64, as in the paper

    first_Brusselator_fom = InstationaryModel(    # here create first FOM (we will change initial cond and T in the following)
        T=time,
        initial_data=u0_vec,
        operator=a_br, # A   
        mass=m_op,     # M
        rhs=None,     # f
        time_stepper=stepper,
        products={
            'l2': m_op,          
            'h1_0_semi': H01_op,  
            'h1': m_op+H01_op          
        }
    )
    

    return first_Brusselator_fom