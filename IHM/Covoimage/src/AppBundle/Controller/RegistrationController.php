<?php

namespace AppBundle\Controller;

use AppBundle\Form\UserRegistrationForm;
use Symfony\Bundle\FrameworkBundle\Controller\Controller;
use AppBundle\Entity\User;
use Sensio\Bundle\FrameworkExtraBundle\Configuration\Route;
use Symfony\Component\HttpFoundation\Request;
class RegistrationController extends Controller
{
    /**
     * @Route("/register", name="user_registration")
     */
    public function registerAction(request $request)
    {

        $user = new User();
        $form = $this->createForm(UserRegistrationForm::class,$user);
        $form->handleRequest($request);
        // À partir du formBuilder, on génère le formulaire
        if ($form->isSubmitted() && $form->isValid()) {
            // Mot de passe crypte automatiquent par le service app.doctrine.hash_password_listener lors de update ou insert
            $em = $this->getDoctrine()->getManager();
            $user->setIsActive(true);
            $em->persist($user);
            $em->flush();
            return $this->get('security.authentication.guard_handler')
                ->authenticateUserAndHandleSuccess(
                    $user,
                    $request,
                    $this->get('app.security.login_form_authenticator'),
                    'main'
                );
        }
        return $this->render('registration/register.html.twig', array(
            'form' => $form->createView(),
        ));
    }
}
