<?php
/**
 * Created by PhpStorm.
 * User: Orkolnium
 * Date: 23/02/2017
 * Time: 17:54
 */

namespace AppBundle\Repository;

use AppBundle\Entity\Trajet;

class TrajetRepository
{
    public function loadValidTrajet($source,$destination,$time){
        /*
         * Pour tout les trajet avec datefin < now
         * ceux qui ont pour ville de depart egale a source ou (une villee tape egale a source avec date <da
         * tefin
         *
         *
         */
        return $this->getEntityManager()
            ->createQueryBuilder(t)
            ->select('t')
            ->from('AppBundle:Trajet','t')
            ->join('AppBundle:Etape','e')
            ->join('AppBundle:Ville','v')
            ->where('v.nomVille = :source AND e.H_etape >= :now')
            ->setParameter('source', $source)
            ->setParameter('dest', $destination)
            ->setParameter('now',$time)
            ->getQuery();
    }
}