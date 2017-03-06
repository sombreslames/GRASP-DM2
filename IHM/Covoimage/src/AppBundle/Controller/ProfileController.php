<?php

namespace AppBundle\Controller;

use AppBundle\Entity\Personne;
use AppBundle\Form\PersonneForm;
use Symfony\Bundle\FrameworkBundle\Controller\Controller;
use AppBundle\Repository\UserRepository;
use Symfony\Component\Routing\Annotation\Route;
use AppBundle\Entity\User;
use Sensio\Bundle\FrameworkExtraBundle\Configuration\Security;
use Symfony\Component\HttpFoundation\Request;
class ProfileController extends Controller
{
    /**
     * @Route("/profile", name="profile")
     * @Security("has_role('ROLE_USER')")
     */
    public function newProfileAction(request $request)
    {
        $pers= new Personne();
        $error=null;
        $em = $this->getDoctrine()->getManager();
        $userid=$this->getUser()->getID_user();
        $pers1=$em->getRepository('AppBundle:Personne')->loadProfile($userid);
        if($pers1){
            return $this->updateProfileAction($pers1);
        }
        $form = $this->createForm(PersonneForm::class,$pers);
        $form->handleRequest($request);
        if ($form->isSubmitted() && $form->isValid()) {
            $pers->setIDPersonne($userid);
            $em->persist($pers);
            $em->flush();
        }

        return $this->render('Profile/FillIn.html.twig', array(
            'form' => $form->createView(),
            'error'         => $error,
        ));
    }
    /**
     * @Route("/modprofile", name="modprofile")
     * @Security("has_role('ROLE_USER')")
     */
    public function updateProfileAction(Personne $personne)
    {

        $formod=$this->createForm(PersonneForm::class,$personne);
        return $this->render('Profile/FillIn.html.twig', array(
            'form' => $formod->createView(),
            'error' =>  null,
        ));
        //HANDLING MODIFY FORM
    }
}
