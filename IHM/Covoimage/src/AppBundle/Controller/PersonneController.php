<?php
namespace AppBundle\Controller;

use Symfony\Bundle\FrameworkBundle\Controller\Controller;
use Symfony\Component\HttpFoundation\Request;
use Sensio\Bundle\FrameworkExtraBundle\Configuration\Route;
use AppBundle\Entity\Personne;
class PersonneController extends Controller
{
	/**
     * Matches /test exactly
     *
     * @Route("/test", name="blog_list")
     */
	public function addAction(Request $request)
	{
		$p1 = new Personne();
		$p1->setNom("Test");
		$p1->setPrenom("Test");
		$p1->setDateNaissance('16-11-1994');
		$p1->setSexe("Homme");
		
		$em = $this->getDoctrine()->getManager();
		$em->persist($p1);
		
		$em->flush();
		if ($request->isMethod('POST')) {
      $request->getSession()->getFlashBag()->add('notice', 'Annonce bien enregistrée.');
      //return $this->redirect($this->generateUrl('oc_platform_view', array('id' => $advert->getId())));
    }

    return $this->render('base.html.twig');
	}
}